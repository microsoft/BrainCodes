# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Variable-length wrapper for ``HBNCachedDataset``.

The cache is built once at the longest length; each batch samples one length
and crops a random sub-window from it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


class MultiLengthHBN(Dataset):
    """Wrap an ``HBNCachedDataset`` built at the longest length and serve cropped
    sub-windows. :class:`OneLengthPerBatchSampler` sets ``current_length_sec`` so
    all items in a batch share the same shape. ``codec_hop_samples`` is the number
    of raw samples per codec token (12 for BioCodec @ 250 Hz).
    """

    def __init__(
        self,
        base: Dataset,
        target_lengths_sec: Sequence[float],
        sr: int = 250,
        codec_hop_samples: int = 12,
    ):
        super().__init__()
        if not getattr(base, "return_raw", False):
            raise ValueError("base dataset must be built with return_raw=True")

        self.base = base
        self.sr = int(sr)
        self.target_lengths_sec = list(target_lengths_sec)
        self.hop = int(codec_hop_samples)

        self.current_length_sec: float = float(max(self.target_lengths_sec))

    def __len__(self) -> int:
        return len(self.base)

    @property
    def coords(self):
        """Forward the per-channel electrode coords from the wrapped dataset."""
        return getattr(self.base, "coords", None)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        codes = item["codes"]  # (C, T_full, K)
        raw = item["raw"]  # (C, T_raw_full)
        T_full = codes.shape[1]

        win_sec = float(self.current_length_sec)
        win_tok = int(round(win_sec * self.sr / self.hop))
        win_samp = win_tok * self.hop  # exact alignment
        if win_tok > T_full or win_samp > raw.shape[-1]:
            win_tok = T_full
            win_samp = raw.shape[-1]
            start_tok = 0
        else:
            start_tok = random.randint(0, T_full - win_tok)
        start_samp = start_tok * self.hop

        return {
            "codes": codes[:, start_tok : start_tok + win_tok],  # (C, T_win, K)
            "raw": raw[:, start_samp : start_samp + win_samp],  # (C, T_win*hop)
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        return {
            "codes": torch.stack([b["codes"] for b in batch], dim=0),
            "raw": torch.stack([b["raw"] for b in batch], dim=0),
        }


class OneLengthPerBatchSampler(BatchSampler):
    """Yield batches of equal length, sampling one length per batch."""

    def __init__(
        self,
        dataset: MultiLengthHBN,
        batch_size: int,
        seed: int = 0,
        weights: Sequence[float] | None = None,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.lengths = list(dataset.target_lengths_sec)
        if weights is None:
            weights = [1.0] * len(self.lengths)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()

    def __iter__(self):
        N = len(self.dataset)
        order = list(range(N))
        self.rng.shuffle(order)
        batches = (
            len(order) // self.batch_size
            if self.drop_last
            else (len(order) + self.batch_size - 1) // self.batch_size
        )
        for b in range(batches):
            idxs = order[b * self.batch_size : (b + 1) * self.batch_size]
            self.dataset.current_length_sec = float(
                self.np_rng.choice(self.lengths, p=self.weights)
            )
            yield idxs

    def __len__(self):
        N = len(self.dataset)
        return (
            N // self.batch_size
            if self.drop_last
            else ((N + self.batch_size - 1) // self.batch_size)
        )
