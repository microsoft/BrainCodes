"""Classification head + fine-tuning utilities for the BrainCodes backbone.

Two training modes:

* ``linear_probe`` — freeze the backbone, train only the head.
* ``full``         — train backbone + head end-to-end (the codec stays frozen).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .backbone import BrainCodes


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class BrainCodesClassifier(nn.Module):
    """Classification head on top of a pretrained :class:`BrainCodes` backbone.

    Both heads read the hourglass bottleneck rather than the full-resolution
    decoder output, whose reconstruction-oriented detail a linear probe tends to
    overfit; every reported result uses a bottleneck tap.

    ``head_type``:

    * ``channel_flat_bottleneck`` — per-channel bottleneck latent → projector →
      flatten → linear classifier. The default, used for most reported results.
    * ``channel_time_flat_bottleneck_noproj`` — the whole bottleneck grid flattened
      straight into a linear classifier, strictly linear in the frozen features.

    Takes pre-tokenized codes ``(B, C, T, K)``; use
    :class:`braincodes.api.BrainCodesModel` to obtain them from raw EEG.
    """

    HEADS = ("channel_flat_bottleneck", "channel_time_flat_bottleneck_noproj")

    def __init__(
        self,
        backbone: BrainCodes,
        num_classes: int,
        hidden: int = 256,
        dropout: float = 0.3,
        head_type: str = "channel_flat_bottleneck",
        n_tokens: int | None = None,
    ):
        super().__init__()
        if head_type not in self.HEADS:
            raise ValueError(
                f"unknown head_type {head_type!r}; expected one of {self.HEADS}"
            )

        self.backbone = backbone
        self.head_type = head_type

        if head_type == "channel_time_flat_bottleneck_noproj":
            if n_tokens is None:
                raise ValueError(
                    "channel_time_flat_bottleneck_noproj needs n_tokens (the "
                    "number of BioCodec frames per window) to size the head"
                )
            bottleneck_tokens = n_tokens
            for f in backbone.backbone.pool_factors:
                bottleneck_tokens = math.ceil(bottleneck_tokens / f)
            self.projector = nn.Identity()
            flat_dim = backbone.D * backbone.C * bottleneck_tokens
        else:
            self.projector = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone.D, hidden),
                nn.GELU(),
            )
            flat_dim = hidden * backbone.C

        self.classifier = nn.Sequential(
            RMSNorm(flat_dim),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, num_classes),
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        out = self.backbone(codes, already_tokenized=True)
        if self.head_type == "channel_time_flat_bottleneck_noproj":
            return self.classifier(out["bottleneck"].flatten(1))
        per_chan = out["bottleneck_per_chan"]  # (B, C, D)
        return self.classifier(self.projector(per_chan).flatten(1))

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode even under a blanket ``.train()``.

        A linear probe is defined on fixed features; re-enabling backbone dropout
        would make them differ between fitting and evaluation.
        """
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        else:
            self.backbone.tokenizer.eval()
        return self


def build_classifier(
    backbone: BrainCodes,
    num_classes: int,
    mode: str = "linear_probe",
    hidden: int = 256,
    dropout: float = 0.3,
    head_type: str = "channel_flat_bottleneck",
    n_tokens: int | None = None,
) -> BrainCodesClassifier:
    """Build a classifier and apply the freezing strategy for ``mode``."""
    if mode not in ("linear_probe", "full"):
        raise ValueError(f"unknown mode {mode!r}")

    model = BrainCodesClassifier(
        backbone,
        num_classes=num_classes,
        hidden=hidden,
        dropout=dropout,
        head_type=head_type,
        n_tokens=n_tokens,
    )

    if mode == "linear_probe":
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model.backbone.eval()
    else:  # full: train the backbone, but keep the BioCodec tokenizer frozen
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        for p in model.backbone.tokenizer.parameters():
            p.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[braincodes] mode={mode}  trainable={n_train:,} / total={n_total:,}")
    return model
