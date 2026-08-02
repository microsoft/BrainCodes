"""HuggingFace-style input preprocessor for raw EEG.

Turns arbitrary raw EEG into the exact tensor BrainCodes expects:

* resampled to the model sample rate (default 250 Hz),
* band-pass + notch filtered,
* channels selected/reordered to match the model montage,
* shaped ``(B, C, T)``.

Typical usage::

    from braincodes import BrainCodesModel, EEGPreprocessor

    model = BrainCodesModel.from_pretrained(["TP9", "AF7", "AF8", "TP10"])
    pre = EEGPreprocessor.from_model(model, notch_freq=60.0)

    eeg = pre(raw, sampling_rate=500, ch_names=raw_ch_names)   # (B, C, T) @ 250 Hz
    emb = model.embed(eeg)
"""

from __future__ import annotations

from math import gcd

import numpy as np
import torch

__all__ = ["EEGPreprocessor"]


class EEGPreprocessor:
    """Resample, filter and channel-align raw EEG for :class:`BrainCodesModel`.

    Parameters
    ----------
    ch_names:
        Target montage (10-05 labels). Output channels follow this order. If a
        call passes its own ``ch_names``, the input is selected/reordered to
        match these.
    sampling_rate:
        Target sample rate in Hz. Inputs at another rate are resampled.
    l_freq, h_freq:
        Band-pass edges in Hz. Set either to ``None`` to make the filter a
        high-/low-pass; set both to ``None`` to skip band-pass filtering.
    notch_freq:
        Power-line frequency to notch out (e.g. ``60.0`` in the US, ``50.0`` in
        the EU). May be a list to notch several harmonics. ``None`` skips it.
    normalize:
        If ``True``, z-score each channel over time after filtering. Leave
        ``False`` and let :meth:`BrainCodesModel.tokenize` normalize instead.
    """

    def __init__(
        self,
        ch_names: list[str],
        sampling_rate: int = 250,
        l_freq: float | None = 0.5,
        h_freq: float | None = 40.0,
        notch_freq: float | list[float] | None = 60.0,
        normalize: bool = False,
    ):
        self.ch_names = list(ch_names)
        self.sampling_rate = int(sampling_rate)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.normalize = bool(normalize)

    @classmethod
    def from_model(cls, model, **overrides) -> EEGPreprocessor:
        """Build a preprocessor bound to ``model``'s montage and sample rate;
        ``overrides`` (e.g. ``notch_freq=50.0``) replace the defaults."""
        cfg = {"ch_names": model.ch_names, "sampling_rate": model.sample_rate}
        cfg.update(overrides)
        return cls(**cfg)

    def __call__(
        self,
        eeg: np.ndarray | torch.Tensor,
        sampling_rate: int | None = None,
        ch_names: list[str] | None = None,
        return_tensors: str = "pt",
    ) -> np.ndarray | torch.Tensor:
        """Preprocess raw EEG ``(C, T)`` or ``(B, C, T)`` -> ``(B, C, T)`` at the
        target sample rate.

        ``sampling_rate`` is the rate of ``eeg`` (default: no resampling).
        ``ch_names`` are its channel labels; if given, channels are selected and
        reordered to the target montage. ``return_tensors`` is ``"pt"`` or ``"np"``.
        """
        x = self._as_numpy(eeg)  # (B, C, T) float64
        if ch_names is not None:
            x = self._reorder_channels(x, ch_names)
        elif x.shape[1] != len(self.ch_names):
            raise ValueError(
                f"expected {len(self.ch_names)} channels; got {x.shape[1]}. "
                "Pass ch_names= so channels can be selected/reordered."
            )

        sr = int(sampling_rate) if sampling_rate is not None else self.sampling_rate
        if sr != self.sampling_rate:
            x = self._resample(x, sr, self.sampling_rate)

        if self.l_freq is not None or self.h_freq is not None:
            x = self._bandpass(x, self.sampling_rate)
        if self.notch_freq is not None:
            x = self._notch(x, self.sampling_rate)

        if self.normalize:
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True)
            x = (x - mu) / np.clip(sd, 1e-6, None)

        if return_tensors == "pt":
            return torch.from_numpy(np.ascontiguousarray(x)).float()
        if return_tensors == "np":
            return x.astype(np.float32)
        raise ValueError(f"return_tensors must be 'pt' or 'np'; got {return_tensors!r}")

    @staticmethod
    def _as_numpy(eeg) -> np.ndarray:
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.detach().cpu().numpy()
        x = np.asarray(eeg, dtype=np.float64)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3:
            raise ValueError(f"expected EEG (B, C, T) or (C, T); got {x.shape}")
        return x

    def _reorder_channels(self, x: np.ndarray, ch_names: list[str]) -> np.ndarray:
        if x.shape[1] != len(ch_names):
            raise ValueError(
                f"ch_names has {len(ch_names)} entries but input has "
                f"{x.shape[1]} channels"
            )
        src = {self._norm(c): i for i, c in enumerate(ch_names)}
        try:
            idx = [src[self._norm(c)] for c in self.ch_names]
        except KeyError as e:
            missing = [c for c in self.ch_names if self._norm(c) not in src]
            raise KeyError(f"channels missing from input: {missing}") from e
        return x[:, idx, :]

    @staticmethod
    def _norm(name: str) -> str:
        return name.rstrip(".").strip().upper()

    @staticmethod
    def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        up, down = sr_out // g, sr_in // g
        return resample_poly(x, up, down, axis=-1)

    def _bandpass(self, x: np.ndarray, fs: int) -> np.ndarray:
        from scipy.signal import butter, filtfilt

        nyq = 0.5 * fs
        if self.l_freq is not None and self.h_freq is not None:
            wn, btype = [self.l_freq / nyq, self.h_freq / nyq], "bandpass"
        elif self.l_freq is not None:
            wn, btype = self.l_freq / nyq, "highpass"
        else:
            wn, btype = self.h_freq / nyq, "lowpass"
        b, a = butter(4, wn, btype=btype)
        return filtfilt(b, a, x, axis=-1)

    def _notch(self, x: np.ndarray, fs: int) -> np.ndarray:
        from scipy.signal import filtfilt, iirnotch

        freqs = self.notch_freq
        if np.isscalar(freqs):
            freqs = [freqs]
        for f0 in freqs:
            if f0 >= 0.5 * fs:
                continue
            b, a = iirnotch(f0, 30.0, fs)
            x = filtfilt(b, a, x, axis=-1)
        return x
