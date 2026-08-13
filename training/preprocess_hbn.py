# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Offline preprocessing for HBN-EEG -> cached ``.npz`` shards for BrainCodes.

Per recording: load EEGLAB ``.set``, pick EEG channels, band-pass 0.5-100 Hz,
notch 60/120 Hz, spline-interpolate flat channels, resample to 250 Hz (BioCodec
rate), robust z-score, cut into ``segment_seconds`` windows, tokenize with frozen
BioCodec.

Run (from the repo root)::

    python -m training.preprocess_hbn \\
        --bids_root data/HBN-EEG/R4 \\
        --out_dir   data/cache/R4 \\
        --keep_subset gsn128_to_64 --keep_raw \\
        --target_sr 250 --segment_seconds 30 --device cuda

Shards hold only signal and electrode coordinates. Participant identifiers stay
in ``manifest.json`` (needed for the leak-free subject split); demographics and
phenotype scores are never read or written.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np
import torch

from braincodes.backbone import BioCodecTokenizer
from braincodes.channels import coords_from_names

from .hbn_dataset import HBNManifestEntry

mne.set_log_level("WARNING")


GSN128_TO_64 = [f"E{i}" for i in range(1, 129, 2)]
CHANNEL_SUBSETS = {
    "gsn128_to_64": GSN128_TO_64,
}


def _robust_zscore(x: np.ndarray, eps: float = 1e-6, clip: float = 20.0) -> np.ndarray:
    """Per-channel median/MAD normalization, clipped to ``+-clip``. ``x`` is ``(C, T)``.

    On a flat/bad channel MAD ~= 0, so the divisor collapses to ``eps`` and
    artifacts would explode to ~1e5 -> +-inf in float16; clipping also bounds
    OOD amplitudes (<0.2% of tokens differ vs unclipped on healthy channels).
    """
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True)
    z = (x - med) / (1.4826 * mad + eps)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def _flag_flat_channels(
    raw, flat_frac: float = 0.01, abs_eps: float = 1e-12
) -> list[str]:
    """Channel names whose MAD is ~0 or a tiny fraction of the array median MAD."""
    x = raw.get_data()
    mad = np.median(np.abs(x - np.median(x, axis=-1, keepdims=True)), axis=-1)
    ref = np.median(mad[np.isfinite(mad)]) if np.isfinite(mad).any() else 0.0
    bad = ~np.isfinite(mad) | (mad <= abs_eps) | (mad < flat_frac * ref)
    bad |= ~np.isfinite(x).all(axis=-1)
    return [c for c, b in zip(raw.ch_names, bad) if b]


def _segment(x: np.ndarray, win: int, overlap: float = 0.0) -> np.ndarray:
    """``(C, T) -> (N, C, win)`` sliding-window segments (drops tail).

    ``overlap`` in [0, 1): fraction of ``win`` shared between windows
    (0.0 = non-overlapping, 0.5 = hop of win // 2).
    """
    C, T = x.shape
    hop = max(1, int(round(win * (1.0 - overlap))))
    if T < win:
        return np.empty((0, C, win), dtype=x.dtype)
    starts = range(0, T - win + 1, hop)
    return np.stack([x[:, s : s + win] for s in starts], axis=0)


@torch.no_grad()
def _tokenize_batched(
    segments: np.ndarray,
    tokenizer: BioCodecTokenizer,
    device: str,
    micro_batch: int = 8,
) -> np.ndarray:
    """``(N, C, T) -> (N, C, T_tok, K)`` int16."""
    outs = []
    x = torch.from_numpy(segments).float()
    for i in range(0, x.shape[0], micro_batch):
        xb = x[i : i + micro_batch].to(device, non_blocking=True)
        codes = tokenizer.tokenize(xb)  # (b, C, T_tok, K)
        outs.append(codes.to("cpu", dtype=torch.int16).numpy())
    return (
        np.concatenate(outs, axis=0)
        if outs
        else np.empty((0, segments.shape[1], 0, tokenizer.n_q), dtype=np.int16)
    )


def _process_recording(
    bdf_path: Path,
    *,
    tokenizer: BioCodecTokenizer,
    target_sr: int,
    segment_seconds: float,
    out_path: Path,
    device: str,
    keep_raw: bool,
    micro_batch: int,
    overlap: float = 0.0,
    keep_channels: list[str] | None = None,
    max_seconds: float | None = None,
    codes_src: Path | None = None,
) -> HBNManifestEntry | None:
    raw = mne.io.read_raw_eeglab(str(bdf_path), preload=True, verbose="ERROR")

    raw.pick("eeg")
    if keep_channels is not None:
        present = [c for c in keep_channels if c in raw.ch_names]
        missing = [c for c in keep_channels if c not in raw.ch_names]
        if missing:
            warnings.warn(
                f"{bdf_path.name}: {len(missing)} requested channels missing, e.g. {missing[:5]}"
            )
        if not present:
            warnings.warn(f"{bdf_path.name}: no requested channels present, skipping")
            return None
        raw.pick(present)
        raw.reorder_channels(present)

    nyq = raw.info["sfreq"] / 2.0
    raw.filter(l_freq=0.5, h_freq=min(100.0, nyq - 1.0), fir_design="firwin")
    notch = [f for f in (60.0, 120.0) if f < nyq - 1.0]
    if notch:
        raw.notch_filter(freqs=notch)

    bads = _flag_flat_channels(raw)
    if bads:
        if len(bads) > 0.5 * len(raw.ch_names):
            warnings.warn(
                f"{bdf_path.name}: {len(bads)}/{len(raw.ch_names)} channels flat, skipping"
            )
            return None
        raw.set_montage("GSN-HydroCel-129", match_case=False, on_missing="warn")
        raw.info["bads"] = bads
        raw.interpolate_bads(method={"eeg": "spline"}, reset_bads=True)

    if raw.info["sfreq"] != target_sr:
        raw.resample(target_sr, npad="auto")

    data = raw.get_data().astype(np.float32)  # (C, T)
    if max_seconds is not None:
        data = data[:, : int(round(max_seconds * target_sr))]
    data = _robust_zscore(data)

    win = int(round(segment_seconds * target_sr))
    segs = _segment(data, win, overlap=overlap)  # (N, C, T)
    if segs.shape[0] == 0:
        warnings.warn(f"no full segments in {bdf_path}")
        return None

    if codes_src is not None:
        if not codes_src.is_file():
            warnings.warn(f"no cached codes at {codes_src}; skipping")
            return None
        with np.load(codes_src) as z:
            codes = np.asarray(z["codes"])
    else:
        codes = _tokenize_batched(segs, tokenizer, device, micro_batch=micro_batch)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = coords_from_names(raw.ch_names, montage="GSN-HydroCel-129")
    payload = {"codes": codes, "coords": coords.astype(np.float32)}
    if keep_raw:
        payload["raw"] = segs.astype(np.float16)
    np.savez_compressed(out_path, **payload)

    entities = _parse_bids_filename(bdf_path.name)
    return HBNManifestEntry(
        path=str(out_path),
        n_segments=int(segs.shape[0]),
        subject=entities.get("sub", ""),
        task=entities.get("task", ""),
        run=entities.get("run", None),
    )


def _parse_bids_filename(name: str) -> dict[str, str]:
    """``sub-XYZ_task-RestingState_run-01_eeg.bdf`` -> dict of entities."""
    stem = name.rsplit(".", 1)[0]
    out: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            out[k] = v
    return out


def _find_recordings(bids_root: Path) -> list[Path]:
    """All ``sub-*/**/eeg/*_eeg.set`` files under a BIDS release root."""
    return sorted(bids_root.glob("sub-*/**/eeg/*_eeg.set"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to biocodec_ckpt.pt; defaults to $BIOCODEC_CKPT.",
    )
    ap.add_argument("--target_sr", type=int, default=250)
    ap.add_argument("--segment_seconds", type=float, default=5.0)
    ap.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Fraction of window overlap between consecutive segments (0.5 = 50%%).",
    )
    ap.add_argument("--n_q", type=int, default=6)
    ap.add_argument("--q_bins", type=int, default=256)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--keep_raw", action="store_true")
    ap.add_argument(
        "--keep_channels",
        nargs="*",
        default=None,
        help="Channels are restricted and reordered to match this list.",
    )
    ap.add_argument(
        "--keep_subset",
        choices=sorted(CHANNEL_SUBSETS.keys()),
        default=None,
        help="Built-in channel subset shortcut (overridden by --keep_channels).",
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional restrict to these task names (BIDS task entity).",
    )
    ap.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Use only the first N seconds of each recording before segmenting.",
    )
    ap.add_argument(
        "--limit_subjects",
        type=int,
        default=None,
        help="Keep all tasks for the first N distinct subjects (sorted).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many recordings (debug).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess and overwrite existing shards instead of reusing them.",
    )
    ap.add_argument(
        "--codes_from",
        type=Path,
        default=None,
        help="Reuse cached codes from this cache dir (skip tokenization).",
    )
    args = ap.parse_args()

    keep_channels = args.keep_channels
    if keep_channels is None and args.keep_subset is not None:
        keep_channels = CHANNEL_SUBSETS[args.keep_subset]
    if keep_channels is not None:
        print(f"keeping {len(keep_channels)} channels (first: {keep_channels[:5]} ...)")

    if args.codes_from is None:
        tokenizer = BioCodecTokenizer(
            ckpt_path=args.ckpt,
            sample_rate=args.target_sr,
            n_q=args.n_q,
            q_bins=args.q_bins,
        ).to(args.device)
        tokenizer.eval()
    else:
        tokenizer = None
        print(f"reusing codes from {args.codes_from}")

    recordings = _find_recordings(args.bids_root)
    if args.tasks:
        wanted = set(args.tasks)
        recordings = [
            p for p in recordings if _parse_bids_filename(p.name).get("task") in wanted
        ]
    if args.limit_subjects:
        seen: list[str] = []
        for p in recordings:
            sub = _parse_bids_filename(p.name).get("sub")
            if sub not in seen:
                seen.append(sub)
        keep_subs = set(seen[: args.limit_subjects])
        recordings = [
            p
            for p in recordings
            if _parse_bids_filename(p.name).get("sub") in keep_subs
        ]
    if args.limit:
        recordings = recordings[: args.limit]
    print(f"found {len(recordings)} EEG recordings under {args.bids_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[HBNManifestEntry] = []

    for i, rec in enumerate(recordings):
        ent = _parse_bids_filename(rec.name)
        sub = ent.get("sub", "unk")
        task = ent.get("task", "unk")
        run = ent.get("run", "")
        suffix = f"_run-{run}" if run else ""
        out_path = args.out_dir / f"sub-{sub}" / f"sub-{sub}_task-{task}{suffix}.npz"
        if out_path.exists() and not args.overwrite:
            try:
                with np.load(out_path) as z:
                    n_seg = int(z["codes"].shape[0])
                manifest.append(
                    HBNManifestEntry(str(out_path), n_seg, sub, task, run or None)
                )
                print(f"[{i+1}/{len(recordings)}] cached {out_path.name}")
                continue
            except Exception as e:
                print(
                    f"[{i+1}/{len(recordings)}] corrupt {out_path.name} ({e}); re-processing"
                )
                out_path.unlink(missing_ok=True)

        codes_src = (
            (args.codes_from / f"sub-{sub}" / out_path.name)
            if args.codes_from
            else None
        )
        entry = _process_recording(
            rec,
            tokenizer=tokenizer,
            target_sr=args.target_sr,
            segment_seconds=args.segment_seconds,
            overlap=args.overlap,
            out_path=out_path,
            device=args.device,
            keep_raw=args.keep_raw,
            micro_batch=args.micro_batch,
            keep_channels=keep_channels,
            max_seconds=args.max_seconds,
            codes_src=codes_src,
        )
        if entry is not None:
            manifest.append(entry)
            print(
                f"[{i+1}/{len(recordings)}] wrote {out_path.name} ({entry.n_segments} segs)"
            )

    manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([asdict(e) for e in manifest], f, indent=2)
    print(
        f"manifest -> {manifest_path}  ({len(manifest)} recordings, "
        f"{sum(e.n_segments for e in manifest)} segments)"
    )


if __name__ == "__main__":
    main()
