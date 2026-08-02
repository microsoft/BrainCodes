"""Injected-oscillation pretext.

Transient narrowband bursts (band-pass-filtered noise centred at ``f``, width
``bandwidth`` -- closer to a physiological rhythm than a pure tone) are injected
into a spatially contiguous region of channels; the model regresses the centre
frequency and burst duration and detects which channels were hit.
"""

from __future__ import annotations

import math

import numpy as np
import torch

EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha_low": (8.0, 10.0),
    "alpha_high": (10.0, 13.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 32.0),
}

DEFAULT_BANDS = ["theta", "alpha_low", "alpha_high", "beta_low", "beta_high"]


def objective_bands(names: list[str] | None = None) -> list[tuple[float, float]]:
    return [EEG_BANDS[n] for n in (names if names is not None else DEFAULT_BANDS)]


@torch.no_grad()
def inject_oscillations(
    x: torch.Tensor,
    sample_rate: float = 250.0,
    bands: list | None = None,
    amp_min: float = 0.1,
    amp_max: float = 0.5,
    dur_range: tuple[float, float] = (2.0, 4.0),
    bandwidth: float = 2.0,
    coords: torch.Tensor | None = None,
    region_neighbor_frac: float = 0.125,
    region_frac: tuple[float, float] = (0.2, 0.4),
    sample_gate: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
):
    """Inject transient narrowband bursts into a spatially contiguous region.

    ``x`` is ``(B, C, T)`` raw EEG at per-channel unit variance. ``bands`` are
    ``[(lo, hi), ...]`` candidates for the centre frequency (default: EEG bands).
    Amplitudes are drawn log-uniformly in ``[amp_min, amp_max]`` std units, so the
    model mostly sees the low-amplitude, hard-to-detect regime. The region covers
    ``region_frac`` of channels, grown by the ``region_neighbor_frac * C`` nearest
    electrodes of each seed.

    Returns ``(x_aug (B,C,T), freq (B,C) Hz [0 off-channel], dur_frac (B,C),
    mask (B,C) bool)``.
    """
    B, C, T = x.shape
    dev = x.device
    g = {"generator": generator} if generator is not None else {}
    win_sec = T / sample_rate

    cds = coords.to(dev)
    cd = torch.cdist(cds, cds)
    k = max(1, min(int(round(region_neighbor_frac * C)), C - 1))
    nn = cd.argsort(dim=1)[:, : k + 1].cpu().numpy()
    flo, fhi = region_frac
    pm = np.zeros((B, C), dtype=bool)
    for b in range(B):
        target = max(1, int(round((flo + (fhi - flo) * np.random.random()) * C)))
        guard = 0
        while pm[b].sum() < target and guard < C:
            pm[b, nn[np.random.randint(C)]] = True
            guard += 1
    mask = torch.from_numpy(pm).to(dev)
    if sample_gate is not None:
        mask = mask & sample_gate.view(B, 1).to(mask.device)

    u_amp = torch.rand(B, C, device=dev, **g)
    A = torch.exp(math.log(amp_min) + (math.log(amp_max) - math.log(amp_min)) * u_amp)

    if bands is None:
        bands = objective_bands()
    edges = torch.tensor(bands, dtype=torch.float32, device=dev)  # (n_bands, 2)
    bi = torch.randint(len(bands), (B, C), device=dev, **g)
    half_bw = bandwidth / 2.0
    lo, hi = edges[bi, 0], edges[bi, 1]
    f = lo + (hi - lo) * torch.rand(B, C, device=dev, **g)  # (B, C) centre freq

    t = torch.arange(T, device=dev, dtype=torch.float32) / sample_rate  # (T,)
    dur = dur_range[0] + (dur_range[1] - dur_range[0]) * torch.rand(
        B, C, 1, device=dev, **g
    )
    start = torch.rand(B, C, 1, device=dev, **g) * (win_sec - dur)
    start = start.clamp_min(0.0)
    u = (t.view(1, 1, T) - start) / dur  # phase within burst (0..1)

    env = torch.where(
        (u >= 0) & (u <= 1),
        0.5 * (1 - torch.cos(2 * torch.pi * u)),
        torch.zeros_like(u),
    )

    noise = torch.randn(B, C, T, device=dev, **g)
    spec = torch.fft.rfft(noise, dim=-1)  # (B, C, Fbins)
    fbins = torch.fft.rfftfreq(T, d=1.0 / sample_rate, device=dev)  # (Fbins,)
    lo_f = (f - half_bw).clamp_min(0.0).unsqueeze(-1)  # (B, C, 1)
    hi_f = (f + half_bw).unsqueeze(-1)
    band = (fbins.view(1, 1, -1) >= lo_f) & (fbins.view(1, 1, -1) <= hi_f)
    carrier = torch.fft.irfft(spec * band, n=T, dim=-1)  # (B, C, T)
    carrier = carrier / (carrier.std(dim=-1, keepdim=True) + 1e-6)

    osc = A.unsqueeze(-1) * env * carrier

    osc = osc * mask.unsqueeze(-1)
    dur_frac = (dur.squeeze(-1) / win_sec) * mask  # (B, C)
    return x + osc, f * mask, dur_frac, mask
