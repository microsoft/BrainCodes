# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Pre-training losses: masked token-CE plus the injection auxiliaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_token_ce(
    logits_list: list[torch.Tensor],
    codes_masked: torch.Tensor,
    return_per_codebook: bool = False,
):
    """Mean masked-slot CE averaged over codebooks. Logits/targets are already
    gathered to the masked positions: ``logits_list`` is list[K] of ``(M, bins)``,
    ``codes_masked`` is ``(M, K)`` long."""
    K = len(logits_list)
    if codes_masked.numel() == 0:
        z = logits_list[0].new_zeros(())
        return (
            (z, logits_list[0].new_full((K,), float("nan")))
            if return_per_codebook
            else z
        )
    per_cb = torch.stack(
        [F.cross_entropy(logits_list[k], codes_masked[:, k]) for k in range(K)]
    )
    mean = per_cb.mean()
    return (mean, per_cb) if return_per_codebook else mean


def injection_loss(out, freq_t, dur_t, mask, fmin, fmax, dmin, dmax):
    """Frequency/duration regression for the injection pretext.

    Scored on injected channels only -- sparse injection makes an all-channel head
    collapse to ~0. Targets are normalized to ``[0, 1]`` by the band range
    ``[fmin, fmax]`` (Hz) and the fractional range ``[dmin, dmax]``. Returns
    ``(freq_loss, dur_loss)``, both zero when nothing is injected.
    """
    freq_n = (freq_t - fmin) / (fmax - fmin)
    dur_n = (dur_t - dmin) / (dmax - dmin)
    if mask.any():
        freq = F.mse_loss(out["freq_pred"][mask], freq_n[mask])
        dur = F.mse_loss(out["dur_pred"][mask], dur_n[mask])
    else:
        freq = out["freq_pred"].new_zeros(())
        dur = out["dur_pred"].new_zeros(())
    return freq, dur


def detect_loss(out, mask):
    """Per-channel injected/not BCE over all channels of samples that received an
    injection. Zero when nothing is injected."""
    has_inj = mask.any(dim=1)  # (B,)
    if has_inj.any():
        return F.binary_cross_entropy_with_logits(
            out["loc_pred"][has_inj], mask[has_inj].float()
        )
    return out["loc_pred"].new_zeros(())
