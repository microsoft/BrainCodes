from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils.data import DataLoader

from .hbn_dataset import HBNCachedDataset, HBNManifestEntry, collate_pretokenized
from .multi_length import MultiLengthHBN, OneLengthPerBatchSampler


def _load_entries(manifest: str | Path | list) -> list[HBNManifestEntry]:
    """Load manifest entries from one path or a list of paths (concatenated).

    Subject IDs are unique per HBN release, so the subject-level split stays
    leak-free when merging releases.
    """
    paths = manifest if isinstance(manifest, (list, tuple)) else [manifest]
    entries: list[HBNManifestEntry] = []
    for p in paths:
        with open(p) as f:
            entries.extend(HBNManifestEntry(**e) for e in json.load(f))
    return entries


def subject_split(
    entries: list[HBNManifestEntry], val_frac: float, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Partition subjects into ``(train, val)``; ``val_frac <= 0`` gives no val."""
    subjects = sorted({e.subject for e in entries})
    if val_frac <= 0:
        return set(subjects), set()

    random.Random(seed).shuffle(subjects)
    n_val = max(1, round(len(subjects) * val_frac))
    return set(subjects[n_val:]), set(subjects[:n_val])


def build_dataloaders(cfg: dict, seed: int = 0) -> tuple[DataLoader, DataLoader | None]:
    """Return ``(train_loader, val_loader)`` from ``cfg["data"]``."""
    dcfg = cfg["data"]
    entries = _load_entries(dcfg.get("manifests") or dcfg["manifest"])
    train_subs, val_subs = subject_split(entries, dcfg.get("val_frac", 0.1), seed)

    bs = dcfg["batch_size"]
    nw = dcfg.get("num_workers", 4)
    lengths_sec = dcfg.get("lengths_sec")

    def _loader(
        subjects: set[str], *, shuffle: bool, drop_last: bool
    ) -> DataLoader | None:
        base = HBNCachedDataset(entries, subjects=subjects, return_raw=True)
        if len(base) == 0:
            return None
        if lengths_sec:
            ds = MultiLengthHBN(
                base,
                target_lengths_sec=lengths_sec,
                sr=cfg["tokenizer"]["sample_rate"],
                codec_hop_samples=dcfg.get("codec_hop_samples", 12),
            )
            sampler = OneLengthPerBatchSampler(
                ds,
                batch_size=bs,
                seed=seed,
                weights=dcfg.get("length_weights"),
                drop_last=drop_last,
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=nw,
                collate_fn=ds.collate,
                pin_memory=True,
                persistent_workers=nw > 0,
            )
        return DataLoader(
            base,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=collate_pretokenized,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

    train_loader = _loader(train_subs, shuffle=True, drop_last=True)
    val_loader = _loader(val_subs, shuffle=False, drop_last=False)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset) if val_loader is not None else 0
    print(
        f"split: {len(train_subs)} train / {len(val_subs)} val subjects | "
        f"{n_train} train / {n_val} val segments"
        + (f" | lengths={lengths_sec}" if lengths_sec else "")
    )

    return train_loader, val_loader
