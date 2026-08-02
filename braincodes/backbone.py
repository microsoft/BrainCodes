"""BrainCodes encoder over frozen BioCodec RVQ tokens.

``BioCodecTokenizer`` codes EEG per channel ``(B,C,t)->(B,C,T_tok,K)``; ``BrainCodes``
embeds and concatenates the K codebooks into ``D_emb=K*d_model`` tokens, projects to
the backbone width and runs a factorized time/channel hourglass transformer.
"""

from __future__ import annotations

import os

import numpy as _np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_biocodec():
    # Deferred so that importing `braincodes` for its architecture alone does
    # not require the separately distributed BioCodec package.
    try:
        from biocodec.model import BioCodecModel
    except ImportError as e:  # pragma: no cover - depends on user environment
        raise ImportError(
            "BrainCodes requires the BioCodec tokenizer, which is distributed "
            "separately:\n\n"
            "    git clone https://github.com/klean2050/BioCodec-release.git\n"
            "    pip install -r BioCodec-release/requirements.txt\n"
            "    export PYTHONPATH=$PWD/BioCodec-release:$PYTHONPATH\n\n"
            "The released BioCodec checkpoint (biocodec_ckpt.pt) ships with that "
            "repository; pass its path as `codec_ckpt` or set $BIOCODEC_CKPT."
        ) from e
    return BioCodecModel


class BioCodecTokenizer(nn.Module):
    """Frozen BioCodec tokenizer applied per-channel by folding C into batch.

    ``ckpt_path`` is required (falling back to ``$BIOCODEC_CKPT``): BrainCodes was
    trained on the codes of one specific BioCodec checkpoint, so an unloaded codec
    emits codes from a random encoder rather than merely degraded ones.
    """

    def __init__(
        self,
        ckpt_path: str | os.PathLike | None = None,
        sample_rate: int = 250,
        n_q: int = 6,
        q_bins: int = 256,
        causal: bool = True,
    ):
        super().__init__()
        BioCodecModel = _import_biocodec()
        codec = BioCodecModel._get_optimized_model(
            sample_rate=sample_rate,
            channels=1,
            causal=causal,
            model_norm="weight_norm",
            signal_normalize=False,
            segment=None,
            name="biocodec_eeg",
            n_q=n_q,
            q_bins=q_bins,
        )
        ckpt_path = ckpt_path or os.environ.get("BIOCODEC_CKPT")
        if not ckpt_path:
            raise ValueError(
                "BioCodec checkpoint is required. Pass codec_ckpt=... or set "
                "$BIOCODEC_CKPT to the biocodec_ckpt.pt shipped with "
                "https://github.com/klean2050/BioCodec-release"
            )
        ckpt_path = str(ckpt_path)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.normpath(os.path.join(_HERE, ckpt_path))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"BioCodec checkpoint not found: {ckpt_path!r}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ck.get("model_state_dict", ck)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        codec.load_state_dict(sd, strict=True)
        self.codec = codec
        self.sample_rate = sample_rate
        self.n_q = n_q
        self.q_bins = q_bins
        self.causal = causal
        for p in self.codec.parameters():
            p.requires_grad_(False)
        self.codec.eval()

    def train(self, mode: bool = True):
        # Keep the codec frozen in eval even under a parent .train().
        super().train(mode)
        self.codec.eval()
        return self

    @property
    def code_dim(self) -> int:
        return self.codec.encoder.dimension

    def codebooks(self) -> list[torch.Tensor]:
        return [layer.codebook for layer in self.codec.quantizer.vq.layers]

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor, micro_batch: int = 0) -> torch.Tensor:
        """``(B, C, t) -> (B, C, T_tok, K)`` long codebook indices.

        ``micro_batch > 0`` chunks the folded ``(B*C, 1, T)`` batch to cap peak
        encoder memory; the codes are identical either way.
        """
        assert x.dim() == 3, f"expected (B,C,t), got {tuple(x.shape)}"
        B, C, T = x.shape
        flat = x.reshape(B * C, 1, T)

        def _encode(chunk: torch.Tensor) -> torch.Tensor:
            frames = self.codec.encode(chunk)  # len-1 list of (codes, scale)
            return torch.cat([f[0] for f in frames], dim=-1)  # (n, K, T_tok)

        if micro_batch and micro_batch < flat.shape[0]:
            codes = torch.cat(
                [
                    _encode(flat[i : i + micro_batch])
                    for i in range(0, flat.shape[0], micro_batch)
                ],
                dim=0,
            )
        else:
            codes = _encode(flat)
        K, T_tok = codes.shape[1], codes.shape[2]
        return codes.view(B, C, K, T_tok).permute(0, 1, 3, 2).contiguous()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """Cached ``(cos, sin)`` tables for rotary position encoding: half-rotation
    ``q' = q·cos + rotate_half(q)·sin`` with ``inv_freq[i] = base^(-2i/head_dim)``
    (Su et al., 2021)."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even per-head dim"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_T = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def forward(self, T: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cached_T >= T
            and self._cached_cos is not None
            and self._cached_cos.device == device
        ):
            return self._cached_cos[:T].to(dtype), self._cached_sin[:T].to(dtype)
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        self._cached_T, self._cached_cos, self._cached_sin = T, cos, sin
        return cos.to(dtype), sin.to(dtype)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with RoPE on ``q``/``k``."""

    def __init__(self, d_model: int, n_heads: int, rope: RoPE, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dH = self.n_heads, self.head_dim
        qkv = (
            self.qkv(x).reshape(B, T, 3, H, dH).permute(2, 0, 3, 1, 4)
        )  # (3, B, H, T, dH)
        q, k, v = qkv.unbind(0)
        cos, sin = self.rope(T, x.device, x.dtype)  # (T, dH)
        cos, sin = cos[None, None], sin[None, None]  # (1, 1, T, dH)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )  # (B, H, T, dH)
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class RoPEEncoderLayer(nn.Module):
    """Pre-LN transformer encoder layer with RoPE attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        rope: RoPE,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, rope, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.ff(self.ln2(x))
        return x


class ChannelRoPEMixed(nn.Module):
    """RoPE-Mixed (Heo et al., 2024) over continuous 3D electrode coordinates.

    Learnable per-head frequency vectors mix the three spatial axes, so each
    (head, rotation-pair) angle is a full 3D dot product ``freq·(x,y,z)`` rather
    than an axis-aligned one. The resulting attention bias depends only on the
    relative displacement between electrodes, making it translation-invariant in
    space and permutation-equivariant across channel ordering.

    forward(coords ``(C, 3)``) -> ``(cos, sin)`` each ``(n_heads, C, head_dim)``.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        n_axes: int = 3,
        base: float = 100.0,
        scale: float = 1.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even per-head dim"
        self.n_heads = n_heads
        self.n_pairs = head_dim // 2
        mag = base ** (-torch.arange(self.n_pairs, dtype=torch.float32) / self.n_pairs)
        dirs = torch.randn(n_heads, self.n_pairs, n_axes)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.freqs = nn.Parameter(dirs * (mag[None, :, None] * scale))

    def forward(self, coords: torch.Tensor):
        coords = coords.to(self.freqs.dtype)
        angle = torch.einsum("hpa,ca->hcp", self.freqs, coords)  # (H, C, n_pairs)
        emb = torch.cat([angle, angle], dim=-1)  # (H, C, head_dim)
        return emb.cos(), emb.sin()


def apply_rope_channel(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate ``x`` ``(B, H, C, head_dim)`` by ``cos``/``sin`` ``(H, C, head_dim)``."""
    return x * cos[None] + _rotate_half(x) * sin[None]


class RoPEChannelSelfAttention(nn.Module):
    """Channel self-attention with RoPE-Mixed applied to ``q``/``k``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        Bt, C, D = x.shape
        H, dH = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(Bt, C, 3, H, dH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (Bt, H, C, dH)
        q = apply_rope_channel(q, cos, sin)
        k = apply_rope_channel(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(out.transpose(1, 2).reshape(Bt, C, D))


class RoPEChannelEncoderLayer(nn.Module):
    """Pre-LN channel encoder layer with RoPE-Mixed attention."""

    def __init__(
        self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEChannelSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), cos, sin))
        x = x + self.ff(self.ln2(x))
        return x


class _TCBlock(nn.Module):
    """``n_time`` time-axis + ``n_chan`` channel-axis layers over ``(B, C, T, D)``.

    Channel position is the ``(cos, sin)`` pair from :class:`ChannelRoPEMixed`,
    re-applied inside every layer rather than added once at the input.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        n_time: int,
        n_chan: int,
        rope: RoPE,
    ):
        super().__init__()
        self.time_tr = nn.Sequential(
            *[
                RoPEEncoderLayer(d_model, n_heads, dim_feedforward, dropout, rope)
                for _ in range(n_time)
            ]
        )
        self.chan_tr = nn.ModuleList(
            [
                RoPEChannelEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_chan)
            ]
        )

    def forward(self, x: torch.Tensor, c_pe) -> torch.Tensor:
        B, C, T, D = x.shape
        xt = self.time_tr(x.reshape(B * C, T, D)).view(B, C, T, D)
        xc = xt.permute(0, 2, 1, 3).reshape(B * T, C, D)
        cos, sin = c_pe
        for layer in self.chan_tr:
            xc = layer(xc, cos, sin)
        return xc.view(B, T, C, D).permute(0, 2, 1, 3)


class _HourglassFactorizedTransformer(nn.Module):
    """``n_blocks`` time/channel blocks with time-axis avg-pool between blocks and
    transpose-conv upsampling + skip-adds back to full resolution.

    ``pool_factors`` has length ``n_blocks - 1``: the factor applied after each
    non-bottleneck block. Compressing the time axis mid-stack forces a fixed-width
    summary through the deepest block, and that bottleneck is what downstream
    heads read.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_blocks: int = 3,
        n_layers_time: int = 1,
        n_layers_chan: int = 1,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        pool_factors: tuple[int, ...] = (2, 2),
        skip_dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        if len(pool_factors) != n_blocks - 1:
            raise ValueError(
                f"pool_factors must have length n_blocks-1={n_blocks - 1}; "
                f"got {len(pool_factors)}"
            )
        ff = dim_feedforward or d_model * 4
        self.pool_factors = list(pool_factors)
        self.skip_drop = nn.Dropout(skip_dropout)
        self.grad_checkpoint = bool(grad_checkpoint)
        rope = RoPE(d_model // n_heads)
        self.blocks = nn.ModuleList(
            [
                _TCBlock(
                    d_model,
                    n_heads,
                    ff,
                    dropout,
                    n_time=n_layers_time,
                    n_chan=n_layers_chan,
                    rope=rope,
                )
                for _ in range(n_blocks)
            ]
        )
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose1d(d_model, d_model, kernel_size=f, stride=f)
                for f in self.pool_factors
            ]
        )
        self.last_bottleneck: torch.Tensor | None = None

    @staticmethod
    def _pool_time(x: torch.Tensor, factor: int) -> torch.Tensor:
        B, C, T, D = x.shape
        y = x.permute(0, 1, 3, 2).reshape(B * C, D, T)
        y = F.avg_pool1d(y, factor, factor, ceil_mode=True)
        return y.reshape(B, C, D, y.shape[-1]).permute(0, 1, 3, 2)

    @staticmethod
    def _up_time(
        x: torch.Tensor, conv: nn.ConvTranspose1d, target_T: int
    ) -> torch.Tensor:
        B, C, T, D = x.shape
        y = conv(x.permute(0, 1, 3, 2).reshape(B * C, D, T))
        if y.shape[-1] > target_T:
            y = y[..., :target_T]
        elif y.shape[-1] < target_T:
            y = F.pad(y, (0, target_T - y.shape[-1]))
        return y.reshape(B, C, D, target_T).permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor, c_pe) -> torch.Tensor:
        n = len(self.blocks)
        skips: list[torch.Tensor] = []
        for i in range(n):
            if self.grad_checkpoint and self.training:
                x = checkpoint(self.blocks[i], x, c_pe, use_reentrant=False)
            else:
                x = self.blocks[i](x, c_pe)
            if i < n - 1:
                skips.append(x)
                x = self._pool_time(x, self.pool_factors[i])
        self.last_bottleneck = x
        for j in range(len(self.pool_factors) - 1, -1, -1):
            skip = skips[j]
            x = self._up_time(x, self.ups[j], skip.shape[2])
            x = x + self.skip_drop(skip)
        return x


class BrainCodes(nn.Module):
    """Encoder over frozen BioCodec RVQ tokens.

    Per-codebook ``d_model`` embeddings are concatenated per ``(b,c,t)`` slot into a
    ``D_emb = K * d_model`` token (``d_model=16`` matches BioCodec, so the embedding
    tables can be initialised from the RVQ codebooks), projected to the backbone
    width, and passed through an hourglass of factorized time/channel blocks.

    Downstream code should read ``bottleneck_per_chan`` -- the deepest, most
    time-compressed representation, and the tap all reported results use.
    """

    def __init__(
        self,
        tokenizer: BioCodecTokenizer,
        n_channels: int,
        d_model: int = 16,
        n_heads: int = 8,
        n_blocks: int = 3,
        n_layers_time: int = 2,
        n_layers_chan: int = 1,
        dropout: float = 0.1,
        init_from_codebooks: bool = True,
        backbone_width: int | None = 128,
        pool_factors: tuple[int, ...] = (2, 2),
        skip_dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.C = n_channels
        self.K = tokenizer.n_q
        self.bins = tokenizer.q_bins
        self.d_model = d_model
        self.D_emb = d_model * self.K
        self.D = int(backbone_width) if backbone_width else self.D_emb
        self.in_proj = (
            nn.Identity() if self.D == self.D_emb else nn.Linear(self.D_emb, self.D)
        )
        self.n_heads = n_heads

        self.code_embs = nn.ModuleList(
            [nn.Embedding(self.bins, d_model) for _ in range(self.K)]
        )

        self.chan_rope = ChannelRoPEMixed(n_heads, self.D // n_heads)
        self.register_buffer("coords", torch.zeros(n_channels, 3), persistent=False)

        self.backbone = _HourglassFactorizedTransformer(
            d_model=self.D,
            n_heads=n_heads,
            n_blocks=n_blocks,
            n_layers_time=n_layers_time,
            n_layers_chan=n_layers_chan,
            dropout=dropout,
            pool_factors=tuple(pool_factors),
            skip_dropout=skip_dropout,
            grad_checkpoint=grad_checkpoint,
        )
        self.norm = nn.LayerNorm(self.D)

        if init_from_codebooks:
            self._init_from_codebooks()

    def set_coords(self, coords) -> None:
        """Register per-channel ``(C, 3)`` electrode coords for the channel PE.

        The channel count may differ from the constructed one: electrodes are
        addressed by position, not by index.
        """
        if isinstance(coords, _np.ndarray):
            coords = torch.from_numpy(coords)
        coords = coords.to(self.coords.device, dtype=self.coords.dtype)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (C, 3); got {tuple(coords.shape)}")
        self.C = coords.shape[0]
        self.coords = coords

    @torch.no_grad()
    def _init_from_codebooks(self) -> None:
        """Seed each code embedding from the matching RVQ codebook vector so the
        backbone starts in the codec's own vector space. Only exact-width codebooks
        are copied; others are left at their random init.
        """
        try:
            cbs = self.tokenizer.codebooks()
        except Exception:
            return
        for i, emb in enumerate(self.code_embs):
            if i >= len(cbs):
                break
            cb = cbs[i]
            if cb.shape == emb.weight.shape:
                emb.weight.copy_(cb.to(emb.weight.dtype))

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, K)`` code indices -> ``(B, C, T, D_emb)`` concatenated embeddings."""
        return torch.cat(
            [emb(codes[..., i]) for i, emb in enumerate(self.code_embs)], dim=-1
        )

    def _pe_coords(self) -> torch.Tensor:
        return self.coords

    def _embed(self, codes: torch.Tensor) -> torch.Tensor:
        return self.in_proj(self.embed_codes(codes))

    def _finalize(self, result: dict, h: torch.Tensor, codes: torch.Tensor) -> dict:
        return result

    def forward(
        self,
        x: torch.Tensor,
        already_tokenized: bool = False,
    ) -> dict:
        """Raw EEG ``(B, C, t)`` or codes ``(B, C, T, K)`` -> representation dict.

        Returns ``tokens`` ``(B, C, T, D)``, ``per_chan`` ``(B, C, D)``,
        ``bottleneck`` ``(B, C, T_b, D)`` and ``bottleneck_per_chan`` ``(B, C, D)``.
        """
        codes = x if already_tokenized else self.tokenizer.tokenize(x)
        codes = codes.long()
        K = codes.shape[-1]
        assert K == self.K, f"shape mismatch: codes={codes.shape}; expected K={self.K}"

        emb = self._embed(codes)
        h = self.norm(self.backbone(emb, self.chan_rope(self._pe_coords())))

        result = {
            "codes": codes,
            "tokens": h,
            "per_chan": h.mean(dim=2),
        }
        bn = self.backbone.last_bottleneck
        if bn is not None:
            bn = self.norm(bn)
            result["bottleneck"] = bn
            result["bottleneck_per_chan"] = bn.mean(dim=2)
        return self._finalize(result, h, codes)
