"""HBN-EEG dataset over pre-tokenized ``.npz`` shards from ``preprocess.py``.

Each shard (one recording) holds:

    codes  : int16   (N, C, T_tok, K)   RVQ indices from frozen BioCodec
    coords : float32 (C, 3)             electrode positions
    raw    : float16 (N, C, T)          only with ``--keep_raw``
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HBNManifestEntry:
    """One row of the dataset manifest = one cached recording."""

    path: str  # absolute path to a .npz shard
    n_segments: int
    subject: str
    task: str
    run: str | None = None


class HBNCachedDataset(Dataset):
    """Random-access dataset over pre-tokenized HBN-EEG segments.

    ``manifest`` is a ``manifest.json`` path or a list of
    :class:`HBNManifestEntry`; ``tasks``/``subjects`` restrict it (the latter
    drives the leak-free train/val split).
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[HBNManifestEntry],
        tasks: Sequence[str] | None = None,
        subjects: Sequence[str] | None = None,
        return_raw: bool = False,
    ):
        if isinstance(manifest, (str, Path)):
            with open(manifest) as f:
                entries = [HBNManifestEntry(**e) for e in json.load(f)]
        else:
            entries = list(manifest)
        if tasks is not None:
            entries = [e for e in entries if e.task in set(tasks)]
        if subjects is not None:
            entries = [e for e in entries if e.subject in set(subjects)]

        self.entries = entries
        self.return_raw = return_raw
        self._index = [
            (i, s) for i, e in enumerate(entries) for s in range(e.n_segments)
        ]
        self._shard_cache: dict[int, np.lib.npyio.NpzFile] = {}
        self._coords: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._index)

    def _open(self, entry_idx: int) -> np.lib.npyio.NpzFile:
        cached = self._shard_cache.get(entry_idx)
        if cached is None:
            cached = self._shard_cache[entry_idx] = np.load(
                self.entries[entry_idx].path, allow_pickle=False, mmap_mode="r"
            )
        return cached

    @property
    def coords(self) -> np.ndarray | None:
        """Per-channel ``(C, 3)`` electrode coords, or None for legacy shards."""
        if self._coords is None and self.entries:
            shard = self._open(0)
            if "coords" in shard.files:
                self._coords = np.asarray(shard["coords"], dtype=np.float32)
        return self._coords

    def __getitem__(self, idx: int) -> dict:
        entry_idx, seg_idx = self._index[idx]
        shard = self._open(entry_idx)

        def _seg(field: str, dtype) -> torch.Tensor:
            return torch.from_numpy(np.asarray(shard[field][seg_idx], dtype=dtype))

        item = {"codes": _seg("codes", np.int64)}
        if self.return_raw:
            item["raw"] = _seg("raw", np.float32)
        return item


def collate_pretokenized(batch: list[dict]) -> dict:
    keys = ["codes"] + (["raw"] if "raw" in batch[0] else [])
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
