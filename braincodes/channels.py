# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Channel-coordinate resolution via MNE standard montages."""

from __future__ import annotations

import mne
import numpy as np


def coords_from_names(
    ch_names: list[str], montage: str = "standard_1005"
) -> np.ndarray:
    """Return ``(C, 3)`` coords, centered and scaled to the unit ball."""
    pos = mne.channels.make_standard_montage(montage).get_positions()
    pos_u = {k.upper(): v for k, v in pos["ch_pos"].items()}

    keys = [ch.rstrip(".").strip().upper() for ch in ch_names]
    missing = [orig for orig, k in zip(ch_names, keys) if k not in pos_u]
    if missing:
        raise KeyError(f"{len(missing)} channels not in montage {montage!r}: {missing}")

    coords = np.stack([np.asarray(pos_u[k]) for k in keys]).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    n = float(np.linalg.norm(coords, axis=1).max())
    return coords / n if n > 0 else coords
