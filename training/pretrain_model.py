# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Pre-training wrapper around the released :class:`BrainCodes` backbone.

Masking, the masked-code heads and the injection auxiliaries live here so that
the released inference path stays free of training-only state.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from braincodes.backbone import BrainCodes


class PretrainBrainCodes(BrainCodes):
    def __init__(
        self,
        *args,
        span_chan_lo: float = 0.10,
        span_chan_hi: float = 0.25,
        span_dur_lo: float = 0.2,
        span_dur_hi: float = 0.8,
        span_neighbor_frac: float = 0.125,
        coord_jitter: float = 0.0,
        inject_on_bottleneck: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        _train_only = {"tokenizer", "skip_dropout", "grad_checkpoint"}
        self._release_arch = {k: v for k, v in kwargs.items() if k not in _train_only}
        self.span_chan_lo = float(span_chan_lo)
        self.span_chan_hi = float(span_chan_hi)
        self.span_dur_lo = float(span_dur_lo)
        self.span_dur_hi = float(span_dur_hi)
        self.span_neighbor_frac = float(span_neighbor_frac)
        self.coord_jitter = float(coord_jitter)
        self.inject_on_bottleneck = bool(inject_on_bottleneck)

        self.mask_emb = nn.Parameter(torch.zeros(self.K, self.d_model))
        nn.init.normal_(self.mask_emb, std=0.02)
        self.heads = nn.ModuleList(
            [nn.Linear(self.D, self.bins) for _ in range(self.K)]
        )
        self.freq_head = nn.Linear(self.D, 1)
        self.dur_head = nn.Linear(self.D, 1)
        self.loc_head = nn.Linear(self.D, 1)

        self._avoid_channels = None
        self._target_codes = None
        self._patch_mask = None
        self._nn_idx = None
        self._nn_for_c = None

    def set_coords(self, coords) -> None:
        super().set_coords(coords)
        self._nn_idx = None
        self._nn_for_c = None
        with torch.no_grad():
            d = torch.cdist(self.coords, self.coords)
            d.fill_diagonal_(float("inf"))
            cap = 0.5 * d.min(dim=1).values.mean()
        self.coord_jitter_cap = cap.reshape(1).to(self.coords.dtype)

    def _pe_coords(self) -> torch.Tensor:
        if not (self.training and self.coord_jitter > 0.0):
            return self.coords
        std = min(self.coord_jitter, float(self.coord_jitter_cap.item()))
        return self.coords + torch.randn_like(self.coords) * std

    def _neighbors(self, C: int):
        if C <= 1 or self.span_neighbor_frac <= 0.0:
            return None
        if self._nn_idx is None or self._nn_for_c != C:
            with torch.no_grad():
                d = torch.cdist(self.coords, self.coords)
                d.fill_diagonal_(float("inf"))
                k = max(1, min(int(round(self.span_neighbor_frac * C)), C - 1))
                self._nn_idx = torch.topk(d, k, largest=False).indices.cpu().numpy()
            self._nn_for_c = C
        return self._nn_idx

    def _mask_pattern(self, codes: torch.Tensor) -> torch.Tensor:
        """Spatially-contiguous span mask: pick a fraction of channels, give each a
        log-uniform time span, and extend every span to that channel's nearest
        electrodes so the model cannot inpaint from an adjacent contact."""
        B, C, T, _ = codes.shape
        avoid = self._avoid_channels
        avoid_np = avoid.detach().cpu().numpy() if avoid is not None else None
        pm = np.zeros((B, C, T), dtype=bool)
        lo, hi = self.span_dur_lo, self.span_dur_hi
        nn_idx = self._neighbors(C)

        for b in range(B):
            allowed = (
                np.where(~avoid_np[b])[0] if avoid_np is not None else np.arange(C)
            )
            if allowed.size == 0:
                continue
            fr = np.random.uniform(self.span_chan_lo, self.span_chan_hi)
            n_ch = max(1, min(int(round(fr * C)), int(allowed.size)))
            for c in np.random.choice(allowed, size=n_ch, replace=False):
                dur = lo * (hi / lo) ** np.random.random()
                L = min(T, max(1, int(round(dur * T))))
                start = int(np.random.randint(0, T - L + 1))
                pm[b, c, start : start + L] = True
                if nn_idx is None:
                    continue
                for nb in nn_idx[c]:
                    if avoid_np is not None and avoid_np[b, nb]:
                        continue
                    pm[b, int(nb), start : start + L] = True

        return torch.from_numpy(pm).to(codes.device)

    def _embed(self, codes: torch.Tensor) -> torch.Tensor:
        B, C, T, K = codes.shape
        patch_mask = self._mask_pattern(codes)
        emb = self.embed_codes(codes).reshape(B, C, T, K, self.d_model)
        mask_e = self.mask_emb.view(1, 1, 1, K, self.d_model).expand_as(emb)
        emb = torch.where(patch_mask[..., None, None], mask_e, emb)
        self._patch_mask = patch_mask
        return self.in_proj(emb.reshape(B, C, T, self.D_emb))

    def _finalize(self, result: dict, h: torch.Tensor, codes: torch.Tensor) -> dict:
        T = codes.shape[2]
        patch_mask = self._patch_mask
        tgt = codes if self._target_codes is None else self._target_codes.long()
        if tgt.shape[2] != T:
            tgt = tgt[:, :, :T]

        result["mask"] = patch_mask
        result["logits"] = [head(h[patch_mask]) for head in self.heads]
        result["codes_masked"] = tgt[patch_mask]

        inj_feat = (
            result["bottleneck_per_chan"]
            if self.inject_on_bottleneck
            else result["per_chan"]
        )
        result["freq_pred"] = self.freq_head(inj_feat).squeeze(-1)
        result["dur_pred"] = self.dur_head(inj_feat).squeeze(-1)
        result["loc_pred"] = self.loc_head(inj_feat).squeeze(-1)
        return result

    def forward(
        self,
        x: torch.Tensor,
        already_tokenized: bool = False,
        target_codes: torch.Tensor | None = None,
        avoid_channels: torch.Tensor | None = None,
    ) -> dict:
        self._avoid_channels = avoid_channels
        self._target_codes = target_codes
        return super().forward(x, already_tokenized=already_tokenized)

    def export_release_weights(self, path: str, step: int = 0) -> dict:
        """Strip pre-training-only tensors and save an inference checkpoint."""
        drop = ("tokenizer.", "heads.", "mask_emb", "freq_head", "dur_head", "loc_head")
        sd = {
            k: v
            for k, v in self.state_dict().items()
            if not any(k.startswith(d) for d in drop)
        }
        ckpt = {
            "model": sd,
            "step": step,
            "arch": self.release_arch(),
            "tokenizer": self.release_tokenizer(),
        }
        torch.save(ckpt, path)
        return ckpt

    def release_arch(self) -> dict:
        return dict(self._release_arch)

    def release_tokenizer(self) -> dict:
        t = self.tokenizer
        return {
            "sample_rate": int(t.sample_rate),
            "n_q": int(t.n_q),
            "q_bins": int(t.q_bins),
            "causal": bool(getattr(t, "causal", True)),
        }
