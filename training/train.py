# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""BrainCodes pre-training.

python -m training.train --config configs/pretrain.yaml
python -m training.train --selftest       # random data
"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from braincodes.backbone import BioCodecTokenizer

from .inject import inject_oscillations, objective_bands
from .loader import build_dataloaders
from .objective import detect_loss, injection_loss, masked_token_ce
from .pretrain_model import PretrainBrainCodes


def build_model(cfg: dict) -> PretrainBrainCodes:
    tok_cfg = cfg["tokenizer"]
    tok = BioCodecTokenizer(
        ckpt_path=tok_cfg["ckpt_path"],
        sample_rate=tok_cfg["sample_rate"],
        n_q=tok_cfg["n_q"],
        q_bins=tok_cfg["q_bins"],
        causal=tok_cfg.get("causal", True),
    )
    m_cfg = cfg["model"]
    return PretrainBrainCodes(
        tokenizer=tok,
        n_channels=cfg["data"]["n_channels"],
        d_model=m_cfg.get("d_model", 16),
        n_heads=m_cfg.get("n_heads", 8),
        n_blocks=m_cfg.get("n_blocks", 3),
        n_layers_time=m_cfg.get("n_layers_time", 2),
        n_layers_chan=m_cfg.get("n_layers_chan", 1),
        dropout=m_cfg.get("dropout", 0.1),
        init_from_codebooks=m_cfg.get("init_from_codebooks", True),
        backbone_width=m_cfg.get("backbone_width", 128),
        pool_factors=tuple(m_cfg.get("pool_factors", (2, 2))),
        span_chan_lo=m_cfg.get("span_chan_lo", 0.10),
        span_chan_hi=m_cfg.get("span_chan_hi", 0.25),
        span_dur_lo=m_cfg.get("span_dur_lo", 0.2),
        span_dur_hi=m_cfg.get("span_dur_hi", 0.8),
        span_neighbor_frac=m_cfg.get("span_neighbor_frac", 0.125),
        coord_jitter=m_cfg.get("coord_jitter", 0.0),
        skip_dropout=m_cfg.get("skip_dropout", 0.0),
        grad_checkpoint=m_cfg.get("grad_checkpoint", False),
        inject_on_bottleneck=m_cfg.get("inject_on_bottleneck", False),
    )


@torch.no_grad()
def _ce_baselines(model, out):
    """Model top-1 against a copy-nearest-unmasked-channel predictor: each masked
    slot is filled from the spatially nearest channel not masked at the same
    instant, i.e. what a pure volume-conduction interpolator would score."""
    tgt = out["codes_masked"]
    if tgt.numel() == 0:
        return None
    K = tgt.shape[1]
    pred = torch.stack([lg.argmax(-1) for lg in out["logits"]], dim=-1)
    top1 = (pred == tgt).float().mean().item()
    codes, mask = out["codes"], out["mask"]
    d = torch.cdist(model.coords, model.coords).to(codes.device)
    hits = []
    for b in range(codes.shape[0]):
        mb = mask[b]
        if not bool(mb.any()) or bool(mb.all()):
            continue
        dm = d.unsqueeze(-1).masked_fill(mb.unsqueeze(0), float("inf"))
        nn = dm.argmin(dim=1)
        cp = torch.gather(codes[b], 0, nn.unsqueeze(-1).expand(-1, -1, K))
        hits.append((cp[mb] == codes[b][mb]).float())
    nn_top1 = torch.cat(hits).mean().item() if hits else float("nan")
    return top1, nn_top1


def cosine_warmup(opt, warmup: int, max_steps: int) -> LambdaLR:
    def lr_fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return LambdaLR(opt, lr_fn)


@torch.no_grad()
def evaluate_injection(
    model,
    loader,
    device,
    sr,
    bands,
    fmin,
    fmax,
    amp_min,
    amp_max,
    dur_min_frac,
    dur_max_frac,
    tok_mb: int = 0,
    bandwidth: float = 2.0,
    region_neighbor_frac: float = 0.125,
    region_frac: tuple[float, float] = (0.2, 0.4),
) -> dict:
    was_training = model.training
    model.eval()

    g = torch.Generator(device=device).manual_seed(29011997)
    agg = {"token": 0.0, "dur": 0.0, "freq": 0.0, "loc": 0.0}
    n = 0
    for batch in loader:
        raw = batch["raw"].to(device, non_blocking=True).float()
        raw = torch.nan_to_num(raw, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        seg_sec = raw.shape[-1] / sr
        dur_range = (
            max(0.5, dur_min_frac * seg_sec),
            max(0.5 + 1e-3, dur_max_frac * seg_sec),
        )
        x_aug, ft, dt, mask = inject_oscillations(
            raw,
            sr,
            bands,
            amp_min=amp_min,
            amp_max=amp_max,
            dur_range=dur_range,
            bandwidth=bandwidth,
            coords=model.coords,
            region_neighbor_frac=region_neighbor_frac,
            region_frac=region_frac,
            generator=g,
        )
        codes = batch["codes"].to(device, non_blocking=True).long()
        if mask.any():
            fresh = model.tokenizer.tokenize(
                x_aug[mask].unsqueeze(1), micro_batch=tok_mb
            ).squeeze(1)
            codes = codes.clone()
            codes[mask] = fresh
        out = model(codes, already_tokenized=True, avoid_channels=mask)

        token = masked_token_ce(out["logits"], out["codes_masked"])
        freq_l, dur_l = injection_loss(
            out, ft, dt, mask, fmin, fmax, dur_min_frac, dur_max_frac
        )
        loc_l = detect_loss(out, mask)
        agg["token"] += float(token)
        agg["dur"] += float(dur_l)
        agg["freq"] += float(freq_l)
        agg["loc"] += float(loc_l)
        n += 1

    if was_training:
        model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def train(
    cfg: dict,
    dataloader,
    device: str = "cuda",
    val_loader=None,
    resume: str | None = None,
) -> None:
    """Run pre-training. ``dataloader`` yields pre-tokenized ``HBNCachedDataset`` dicts."""
    torch.set_float32_matmul_precision("high")

    model = build_model(cfg)
    model.to(device).train()

    coords = getattr(dataloader.dataset, "coords", None)
    if coords is None:
        raise RuntimeError("dataset has no coords; you have to preprocess again")
    model.set_coords(coords)
    print(f"[train] coords set: shape={tuple(coords.shape)}")
    base_coords = torch.as_tensor(coords, dtype=torch.float32)
    rand_chan = cfg["data"].get("randomize_channels")
    if rand_chan:
        _rc_min = int(rand_chan.get("min", 16))
        _rc_max = int(rand_chan.get("max", base_coords.shape[0]))
        print(f"[train] randomize_channels ON: n in [{_rc_min},{_rc_max}] per batch")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    sched = cosine_warmup(opt, cfg["optim"]["warmup_steps"], cfg["optim"]["max_steps"])
    accum = int(cfg["optim"].get("accum_steps", 1))

    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if resume is not None:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        model_keys = set(model.state_dict().keys())

        def _norm(k: str) -> str:
            return k.replace("_orig_mod.", "")

        remap = {_norm(k): k for k in model_keys}
        sd = {}
        for k, v in ck["model"].items():
            target = remap.get(_norm(k))
            if target is not None:
                sd[target] = v
        model.load_state_dict(sd, strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_step = int(ck.get("step", 0))
        print(f"[resume] loaded {resume}  step={start_step}")

    max_steps = cfg["optim"]["max_steps"]
    log_every = cfg["logging"]["log_every"]
    ckpt_every = cfg["logging"]["ckpt_every"]
    val_every = cfg["logging"].get("val_every", ckpt_every)

    inj = cfg["objective"]["inject"]
    inj_bands = objective_bands(cfg["objective"].get("bands"))
    inj_fmin = min(b[0] for b in inj_bands)
    inj_fmax = max(b[1] for b in inj_bands)
    sr = cfg["tokenizer"]["sample_rate"]
    w_tok = cfg["objective"].get("token_weight", 1.0)
    tok_mb = cfg["tokenizer"].get("tokenize_micro_batch", 0)
    inject_frac = float(inj.get("inject_frac", 0.4))
    region_neighbor_frac = float(inj.get("region_neighbor_frac", 0.125))
    region_frac = (
        float(inj.get("region_frac_lo", 0.2)),
        float(inj.get("region_frac_hi", 0.4)),
    )
    print(
        f"[train] recon/inject split: "
        f"{100*(1-inject_frac):.0f}% recon / {100*inject_frac:.0f}% inject "
        f"| region_frac={region_frac} nn={region_neighbor_frac}"
    )

    amp = str(cfg["optim"].get("amp", "fp32")).lower()
    use_amp = amp == "bf16" and device.startswith("cuda")
    if amp not in ("bf16", "fp32"):
        raise ValueError(f"optim.amp must be 'bf16' or 'fp32', got {amp!r}")

    def autocast():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )

    print(f"[train] precision: {'bf16 autocast' if use_amp else 'fp32'}")

    step = start_step
    micro = 0
    micro_agg: dict[str, float] = {}
    opt.zero_grad(set_to_none=True)
    done = False
    while not done:
        for batch in dataloader:
            if rand_chan:
                _n = int(torch.randint(_rc_min, _rc_max + 1, (1,)).item())
                _idx = torch.randperm(base_coords.shape[0])[:_n].sort().values
                batch["raw"] = batch["raw"][:, _idx]
                batch["codes"] = batch["codes"][:, _idx]
                model.set_coords(base_coords[_idx])
            raw = batch["raw"].to(device, non_blocking=True).float()
            raw = torch.nan_to_num(raw, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
            seg_sec = raw.shape[-1] / sr
            dmin_f = inj.get("dur_min_frac", 0.1)
            dmax_f = inj.get("dur_max_frac", 0.5)
            dur_range = (
                max(0.5, dmin_f * seg_sec),
                max(0.5 + 1e-3, dmax_f * seg_sec),
            )
            is_inject = torch.rand(raw.shape[0], device=device) < inject_frac
            x_inj, ft, dt, mask = inject_oscillations(
                raw,
                sr,
                inj_bands,
                amp_min=inj.get("amp_min", 0.1),
                amp_max=inj.get("amp_max", 0.5),
                dur_range=dur_range,
                bandwidth=inj.get("bandwidth", 2.0),
                coords=model.coords,
                region_neighbor_frac=region_neighbor_frac,
                region_frac=region_frac,
                sample_gate=is_inject,
            )
            # Re-tokenize only the changed channels (the codec is per-channel
            # independent, so cached clean codes stay valid elsewhere).
            cache = batch["codes"].to(device, non_blocking=True).long()
            target_codes = cache.clone()
            if mask.any():
                fresh = model.tokenizer.tokenize(
                    x_inj[mask].unsqueeze(1), micro_batch=tok_mb
                ).squeeze(1)
                assert fresh.shape[-2] == cache.shape[-2], (
                    f"T_tok mismatch: fresh={fresh.shape[-2]} "
                    f"cached={cache.shape[-2]}"
                )
                target_codes[mask] = fresh
            tokenized = target_codes
            avoid = is_inject.view(-1, 1).expand(-1, tokenized.shape[1])
            with autocast():
                out = model(
                    tokenized,
                    already_tokenized=True,
                    target_codes=target_codes,
                    avoid_channels=avoid,
                )
                if out["codes_masked"].numel() > 0:
                    token, token_per_cb = masked_token_ce(
                        out["logits"],
                        out["codes_masked"],
                        return_per_codebook=True,
                    )
                else:
                    token = raw.new_zeros(())
                    token_per_cb = raw.new_zeros(model.K)
                if mask.any():
                    freq_l, dur_l = injection_loss(
                        out,
                        ft,
                        dt,
                        mask,
                        inj_fmin,
                        inj_fmax,
                        dmin_f,
                        dmax_f,
                    )
                    loc_l = detect_loss(out, mask)
                else:
                    z = raw.new_zeros(())
                    freq_l, dur_l, loc_l = z, z, z
                loss = (
                    w_tok * token
                    + inj.get("dur_weight", 1.0) * dur_l
                    + inj.get("freq_weight", 1.0) * freq_l
                    + inj.get("loc_weight", 1.0) * loc_l
                )

                loss_dict = {
                    "loss": loss,
                    "token": token.detach(),
                    "token_per_cb": token_per_cb.detach(),
                    "dur": dur_l.detach(),
                    "freq": freq_l.detach(),
                    "loc": loc_l.detach(),
                }

            (loss_dict["loss"] / accum).backward()
            for _k in ("loss", "token", "dur", "freq", "loc"):
                if _k in loss_dict:
                    micro_agg[_k] = micro_agg.get(_k, 0.0) + float(
                        loss_dict[_k].detach()
                    )
            micro += 1
            if micro % accum != 0:
                continue
            eff = {k: v / accum for k, v in micro_agg.items()}
            micro_agg = {}
            torch.nn.utils.clip_grad_norm_(trainable, cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if step and step % log_every == 0:
                extra = (
                    f" dur={eff['dur']:.3f} freq={eff['freq']:.3f} loc={eff['loc']:.3f}"
                )
                bl = _ce_baselines(model, out)
                if bl is not None:
                    extra += f" | top1={bl[0]:.3f} vs copy_nn={bl[1]:.3f}"
                print(
                    f"step={step:6d} lr={sched.get_last_lr()[0]:.2e} "
                    f"loss={eff['loss']:.3f} tok={eff['token']:.3f}{extra}"
                )
            if val_loader is not None and step > 0 and step % val_every == 0:
                if rand_chan:
                    model.set_coords(base_coords)
                v = evaluate_injection(
                    model,
                    val_loader,
                    device,
                    sr,
                    inj_bands,
                    inj_fmin,
                    inj_fmax,
                    inj.get("amp_min", 0.1),
                    inj.get("amp_max", 0.5),
                    inj.get("dur_min_frac", 0.1),
                    inj.get("dur_max_frac", 0.5),
                    tok_mb=tok_mb,
                    bandwidth=inj.get("bandwidth", 2.0),
                    region_neighbor_frac=region_neighbor_frac,
                    region_frac=region_frac,
                )
                print(
                    f"           val | tok={v['token']:.3f} "
                    f"dur={v['dur']:.3f} freq={v['freq']:.3f} loc={v['loc']:.3f}"
                )
            if step and step % ckpt_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step,
                    },
                    save_dir / f"ckpt_{step:07d}.pt",
                )
            step += 1
            if step >= max_steps:
                done = True
                break


def _selftest() -> None:
    cfg = {
        "tokenizer": {
            "ckpt_path": None,
            "sample_rate": 250,
            "n_q": 6,
            "q_bins": 256,
            "causal": True,
        },
        "data": {"n_channels": 19},
        "model": {
            "d_model": 16,
            "n_heads": 8,
            "n_blocks": 3,
            "dropout": 0.1,
            "init_from_codebooks": True,
        },
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, t = 2, 19, 250 * 5

    model = build_model(cfg)
    coords = torch.randn(C, 3)
    model.set_coords(coords / coords.norm(dim=1, keepdim=True).clamp(min=1e-6))
    model.to(device).train()

    out = model(torch.randn(B, C, t, device=device))
    token = masked_token_ce(out["logits"], out["codes_masked"])
    token.backward()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"selftest ok | T_tok={out['codes'].shape[2]} tok={token.item():.3f} "
        f"| trainable={n_train:,} frozen={n_frozen:,}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--selftest", action="store_true")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: cuda if available else cpu).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a ckpt to resume from. Restores model + opt + scheduler + step.",
    )
    args = p.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.config is None:
        raise SystemExit("provide --config or --selftest")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg, seed=cfg.get("seed", 0))
    train(
        cfg,
        train_loader,
        device=device,
        val_loader=val_loader,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
