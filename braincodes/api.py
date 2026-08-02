"""High-level API for the BrainCodes EEG foundation model.

Typical usage::

    import torch
    from braincodes import BrainCodesModel

    ch_names = ["TP9", "AF7", "AF8", "TP10"]          # your montage (10-05 names)
    model = BrainCodesModel.from_pretrained(ch_names, device="cuda")

    eeg = torch.randn(8, 4, 1000)                       # (B, C, T) @ 250 Hz
    emb = model.embed(eeg)                              # (B, C*D) channel embeddings

    clf = model.build_classifier(num_classes=3, mode="linear_probe")
    logits = clf(model.tokenize(eeg))                   # (B, num_classes)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .backbone import BioCodecTokenizer, BrainCodes
from .channels import coords_from_names
from .classifier import BrainCodesClassifier, build_classifier

DEFAULT_ARCH = {
    "d_model": 16,
    "n_heads": 8,
    "n_blocks": 3,
    "n_layers_time": 2,
    "n_layers_chan": 1,
    "dropout": 0.1,
    "init_from_codebooks": True,
    "backbone_width": 128,
    "pool_factors": (2, 2),
}
DEFAULT_TOKENIZER = {"sample_rate": 250, "n_q": 6, "q_bins": 256, "causal": True}

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "braincodes.pt"


class BrainCodesModel:
    """Pretrained BrainCodes backbone bound to a fixed electrode montage.

    Create one instance per montage with :meth:`from_pretrained`. The instance
    tokenizes raw EEG, extracts embeddings, and builds classification heads for
    linear-probe / full fine-tuning.
    """

    def __init__(self, backbone: BrainCodes, ch_names: list[str], device: str = "cpu"):
        self.backbone = backbone.to(device).eval()
        self.ch_names = list(ch_names)
        self.device = device
        self.sample_rate = backbone.tokenizer.sample_rate

    @classmethod
    def from_pretrained(
        cls,
        ch_names: list[str],
        weights: str | os.PathLike | None = None,
        device: str = "cpu",
        montage: str = "standard_1005",
        codec_ckpt: str | os.PathLike | None = None,
        arch: dict | None = None,
        tokenizer_cfg: dict | None = None,
    ) -> BrainCodesModel:
        """Load the released weights for a montage given by ``ch_names``.

        ``ch_names`` are 10-05 electrode labels (e.g. ``["TP9", "AF7", ...]``);
        their count defines the number of input channels. ``codec_ckpt`` points
        at the BioCodec checkpoint (see README); it falls back to
        ``$BIOCODEC_CKPT``.
        """
        a = {**DEFAULT_ARCH, **(arch or {})}
        t = {**DEFAULT_TOKENIZER, **(tokenizer_cfg or {})}
        weights = Path(weights) if weights is not None else _DEFAULT_WEIGHTS
        if not Path(weights).is_file():
            raise FileNotFoundError(f"weights not found: {weights}")

        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("arch"), dict):
            a = {**a, **ckpt["arch"], **(arch or {})}
        if isinstance(ckpt, dict) and isinstance(ckpt.get("tokenizer"), dict):
            t = {**t, **ckpt["tokenizer"], **(tokenizer_cfg or {})}

        tokenizer = BioCodecTokenizer(ckpt_path=codec_ckpt, **t)
        backbone = BrainCodes(
            tokenizer=tokenizer,
            n_channels=len(ch_names),
            d_model=a["d_model"],
            n_heads=a["n_heads"],
            n_blocks=a["n_blocks"],
            n_layers_time=a["n_layers_time"],
            n_layers_chan=a["n_layers_chan"],
            dropout=a["dropout"],
            init_from_codebooks=a["init_from_codebooks"],
            backbone_width=a["backbone_width"],
            pool_factors=tuple(a["pool_factors"]),
        )

        sd = ckpt.get("model", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        sd = {
            k: v
            for k, v in sd.items()
            if not k.startswith(("tokenizer.", "heads."))
            and k != "mask_emb"
            and not k.split(".")[0].endswith(("_head",))
        }
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("tokenizer.")]
        if missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch loading {weights}:\n"
                f"  missing={missing[:8]}\n  unexpected={unexpected[:8]}"
            )

        backbone.set_coords(coords_from_names(ch_names, montage=montage))
        return cls(backbone, ch_names, device=device)

    @torch.no_grad()
    def tokenize(
        self, eeg: torch.Tensor | np.ndarray, normalize: bool = True
    ) -> torch.Tensor:
        """Raw EEG ``(B, C, T)`` @ ``sample_rate`` → codes ``(B, C, T_tok, K)``.

        ``normalize`` z-scores each channel-window (matching training). Input
        should already be band-pass filtered and resampled to ``sample_rate``.
        """
        x = self._as_tensor(eeg)
        if normalize:
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
            x = (x - mu) / sd
        return self.backbone.tokenizer.tokenize(x)

    @torch.no_grad()
    def embed(
        self,
        eeg: torch.Tensor | np.ndarray,
        normalize: bool = True,
        pool: str = "channel",
    ) -> torch.Tensor:
        """Extract frozen embeddings from raw EEG ``(B, C, T)``.

        ``pool``:

        * ``"channel"`` — per-channel bottleneck latent flattened to ``(B, C*D)``.
        * ``"mean"``    — mean over channels of the bottleneck, ``(B, D)``.
        * ``"none"``    — full bottleneck grid ``(B, C, T_b, D)``.
        * ``"tokens"``  — full-resolution decoder output ``(B, C, T_tok, D)``.

        Every reported result is measured on the bottleneck; ``"tokens"`` is
        reconstruction-oriented and exposed only for analysis.
        """
        if pool not in ("channel", "mean", "none", "tokens"):
            raise ValueError(f"unknown pool {pool!r}")
        codes = self.tokenize(eeg, normalize=normalize)
        out = self.backbone(codes, already_tokenized=True)
        if pool == "tokens":
            return out["tokens"]
        if pool == "channel":
            return out["bottleneck_per_chan"].flatten(1)
        if pool == "mean":
            return out["bottleneck_per_chan"].mean(dim=1)
        return out["bottleneck"]

    def build_classifier(
        self,
        num_classes: int,
        mode: str = "linear_probe",
        hidden: int = 256,
        dropout: float = 0.3,
        head_type: str = "channel_flat_bottleneck",
        n_times: int | None = None,
    ) -> BrainCodesClassifier:
        """Attach a classification head for ``mode`` (linear_probe / full).

        The returned module consumes tokenized codes: ``clf(model.tokenize(eeg))``.
        ``n_times`` (samples per window) is required only for
        ``head_type="channel_time_flat_bottleneck_noproj"``, whose input size
        depends on the window length.
        """
        n_tokens = None
        if n_times is not None:
            probe = torch.zeros(1, len(self.ch_names), int(n_times), device=self.device)
            with torch.no_grad():
                n_tokens = self.tokenize(probe).shape[2]
        elif head_type == "channel_time_flat_bottleneck_noproj":
            raise ValueError(
                "head_type='channel_time_flat_bottleneck_noproj' needs n_times "
                "(samples per window) to size the head; pass n_times=eeg.shape[-1]."
            )
        clf = build_classifier(
            self.backbone,
            num_classes=num_classes,
            mode=mode,
            hidden=hidden,
            dropout=dropout,
            head_type=head_type,
            n_tokens=n_tokens,
        )
        return clf.to(self.device)

    def _as_tensor(self, eeg) -> torch.Tensor:
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        eeg = eeg.float().to(self.device)
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        if eeg.dim() != 3:
            raise ValueError(
                f"expected EEG (B, C, T) or (C, T); got {tuple(eeg.shape)}"
            )
        if eeg.shape[1] != len(self.ch_names):
            raise ValueError(
                f"expected {len(self.ch_names)} channels; got {eeg.shape[1]}"
            )
        return eeg
