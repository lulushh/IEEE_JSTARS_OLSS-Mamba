"""Patch-based LST-Mamba for hyperspectral image classification.

This file implements the method discussed in the conversation in a clean,
patch-based form.  The model takes one local hyperspectral patch and predicts the
label of the center pixel.

Input/output convention
-----------------------
    input  patch : [B, P, P, C]
        B: mini-batch size
        P: spatial patch size, normally 9/11/13
        C: number of spectral bands

    output logits: [B, num_classes]
        Each row is the class prediction for the center pixel of the patch.

Core ideas
----------
1. Low-rank spectral-spatial TokenLearner
   A full 3-D token attention tensor A_k(c,h,w) is expensive for HSI data.
   Instead, every token uses a low-rank factorized attention:

       A_k(c,h,w) ~= sum_r A^c_{k,r}(c) * A^s_{k,r}(h,w)

   The spatial factor A^s is predicted from local stem features.  The spectral
   factor A^c is predicted from the patch mean spectrum and biased by ordered
   spectral anchors.  This produces a small number of joint spectral-spatial
   tokens before Mamba sees the data.

2. Order-aware Tiny Mamba
   TokenLearner tokens are learned summaries, but Mamba is a sequence/state
   model.  The spectral anchor prior gives token k a low-to-high spectral
   preference, so the token sequence has a meaningful order.  A tiny Mamba block
   then models state transitions among the ordered spectral-spatial tokens.

3. Center-pixel classification
   HSI patch classification usually labels the center pixel.  The final
   classifier fuses the center local feature with the Mamba-modeled token
   summary.  The auxiliary center-consistency loss in ``utils/losses.py`` keeps
   the global token summary aligned with the target pixel's local feature.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Official implementation from the mamba-ssm package.
    from mamba_ssm import Mamba as _Mamba
except Exception:  # pragma: no cover - used only when mamba-ssm is unavailable.
    _Mamba = None


class _FallbackSequenceMixer(nn.Module):
    """CPU-safe fallback used only for smoke tests without mamba-ssm.

    The paper/model should be reported with real Mamba from ``mamba-ssm``.  This
    GRU fallback exists only so that the repository can be imported and tested on
    machines that do not have the fused Mamba CUDA kernels installed.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        warnings.warn(
            "mamba_ssm is not available; using a GRU fallback for smoke tests. "
            "Install mamba-ssm for final LST-Mamba experiments.",
            RuntimeWarning,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.proj(y)


class TinyMambaBlock(nn.Module):
    """Residual Mamba block for short learned-token sequences.

    Sequence length is ``K`` learned tokens rather than ``P*P`` pixels or ``C``
    bands.  Therefore the block is intentionally small.  Bidirectional processing
    is optional; when enabled, one Mamba scans low-to-high token order and the
    other scans high-to-low order, then the two outputs are averaged.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.norm1 = nn.LayerNorm(dim)

        if _Mamba is not None:
            self.mamba_fwd = _Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = (
                _Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                if bidirectional
                else None
            )
        else:
            self.mamba_fwd = _FallbackSequenceMixer(dim)
            self.mamba_bwd = _FallbackSequenceMixer(dim) if bidirectional else None

        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, K, D]
        x = self.norm1(tokens)
        y = self.mamba_fwd(x)
        if self.mamba_bwd is not None:
            # Reverse-token state propagation gives complementary high-to-low
            # spectral-state flow while preserving the original token order.
            y_bwd = torch.flip(self.mamba_bwd(torch.flip(x, dims=[1])), dims=[1])
            y = 0.5 * (y + y_bwd)
        tokens = tokens + self.drop(y)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class LightweightPatchStem(nn.Module):
    """Cheap local spectral-spatial feature extractor for one patch.

    A heavy 3-D CNN is avoided.  The stem first projects raw bands into a compact
    feature dimension, then applies a depthwise spatial 3x3 convolution to inject
    local neighborhood information with very few parameters.
    """

    def __init__(self, in_bands: int, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_bands, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x_chw: torch.Tensor) -> torch.Tensor:
        # x_chw: [B, C, P, P]
        return self.net(x_chw)


class LowRankSpectralSpatialTokenLearner(nn.Module):
    """Low-rank TokenLearner for HSI patches.

    The module compresses a patch into ``K`` joint spectral-spatial tokens.  For
    each token and each rank component, the attention is factorized into:

    - spatial factor A_s: where to look inside the local patch;
    - spectral factor A_c: which spectral bands or band regions to emphasize.

    Spectral anchors are added to the spectral logits so token 1 prefers lower
    bands and token K prefers higher bands.  This provides an explicit order for
    the downstream Mamba state model.
    """

    def __init__(
        self,
        in_bands: int,
        dim: int,
        num_tokens: int = 8,
        rank: int = 1,
        min_sigma: float = 0.08,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_bands = int(in_bands)
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.rank = int(rank)
        self.min_sigma = float(min_sigma)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, num_tokens * rank, kernel_size=1, bias=True),
        )

        hidden = max(16, dim // 2)
        self.spectral_router = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, num_tokens * rank, kernel_size=1, bias=True),
        )

        # Band-value embedding transforms a weighted raw spectrum into the common
        # hidden dimension.  It helps preserve original spectral responses rather
        # than relying only on stem features.
        self.band_value_embed = nn.Parameter(torch.empty(in_bands, dim))
        nn.init.trunc_normal_(self.band_value_embed, std=0.02)

        band_pos = torch.linspace(0.0, 1.0, in_bands)
        token_mu = torch.linspace(0.0, 1.0, num_tokens)
        self.register_buffer("band_pos", band_pos, persistent=False)
        self.register_buffer("token_mu", token_mu, persistent=False)
        self.log_sigma = nn.Parameter(torch.full((num_tokens,), math.log(0.18)))

        self.token_norm = nn.LayerNorm(dim)
        self.token_proj = nn.Linear(dim, dim)

    def _spectral_anchor_log_prior(self) -> torch.Tensor:
        """Gaussian log-prior [K, C] that defines ordered spectral anchors."""
        sigma = F.softplus(self.log_sigma) + self.min_sigma
        diff = self.band_pos[None, :] - self.token_mu[:, None]
        return -0.5 * (diff / sigma[:, None]).pow(2)

    def forward(self, x_chw: torch.Tensor, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_chw: [B, C, P, P], feat: [B, D, P, P]
        b, c, p_h, p_w = x_chw.shape
        k, r = self.num_tokens, self.rank

        # Spatial factor: normalize over all patch locations.
        spatial_logits = self.spatial_attn(feat).view(b, k, r, p_h * p_w)
        spatial_attn = F.softmax(spatial_logits, dim=-1).view(b, k, r, p_h, p_w)

        # Spectral factor: descriptor is the patch mean spectrum, so the spectral
        # attention is sample-adaptive but still extremely lightweight.
        spectral_descriptor = x_chw.mean(dim=(2, 3)).unsqueeze(1)  # [B, 1, C]
        spectral_logits = self.spectral_router(spectral_descriptor).view(b, k, r, c)
        spectral_logits = spectral_logits + self._spectral_anchor_log_prior()[None, :, None, :]
        spectral_attn = F.softmax(spectral_logits, dim=-1)  # [B, K, R, C]

        # Spatial token term: summarize stem feature maps.
        spatial_tokens = torch.einsum("bkrhw,bdhw->bkrd", spatial_attn, feat)

        # Spectral token term: summarize raw spectra, then map bands to D-dim
        # values through a learnable band-value table.
        spectral_response = torch.einsum("bkrc,bchw->bkrhw", spectral_attn, x_chw)
        spectral_response = torch.einsum("bkrhw,bkrhw->bkr", spectral_response, spatial_attn)
        spectral_values = torch.einsum("bkrc,cd->bkrd", spectral_attn, self.band_value_embed)
        spectral_tokens = spectral_response[..., None] * spectral_values

        tokens = (spatial_tokens + spectral_tokens).sum(dim=2)  # [B, K, D]
        tokens = self.token_proj(self.token_norm(tokens))

        aux = {
            "spatial_attn": spatial_attn.mean(dim=2),      # [B, K, P, P]
            "spectral_attn": spectral_attn.mean(dim=2),    # [B, K, C]
            "tokens_before_mamba": tokens,
        }
        return tokens, aux


@dataclass
class LSTMambaConfig:
    """Configuration container used by experiment scripts and profilers."""

    in_bands: int
    num_classes: int
    patch_size: int = 11
    dim: int = 64
    num_tokens: int = 8
    rank: int = 1
    mamba_depth: int = 1
    d_state: int = 8
    d_conv: int = 3
    expand: int = 1
    dropout: float = 0.05
    bidirectional: bool = True


class LSTMambaPatchClassifier(nn.Module):
    """Patch-based center-pixel classifier.

    The model predicts one label per input patch.  It is therefore compatible
    with standard HSI patch classification and can still produce full-scene maps
    by sliding/batching patches over the whole image at test time.
    """

    def __init__(self, cfg: LSTMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = LightweightPatchStem(cfg.in_bands, cfg.dim, dropout=cfg.dropout)
        self.token_learner = LowRankSpectralSpatialTokenLearner(
            in_bands=cfg.in_bands,
            dim=cfg.dim,
            num_tokens=cfg.num_tokens,
            rank=cfg.rank,
        )
        self.mamba_blocks = nn.Sequential(
            *[
                TinyMambaBlock(
                    dim=cfg.dim,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                    bidirectional=cfg.bidirectional,
                )
                for _ in range(cfg.mamba_depth)
            ]
        )

        # Center feature is the local descriptor at the target pixel.  Token
        # summary is the global patch descriptor after Mamba.  Their fusion gives
        # the classifier both target-specific and contextual information.
        self.fusion = nn.Sequential(
            nn.LayerNorm(cfg.dim * 2),
            nn.Linear(cfg.dim * 2, cfg.dim),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim, cfg.dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(cfg.dim, cfg.num_classes)

    def forward(self, patch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # patch: [B, P, P, C]
        if patch.ndim != 4:
            raise ValueError(f"Expected patch [B,P,P,C], got shape {tuple(patch.shape)}")
        b, p_h, p_w, c = patch.shape
        if c != self.cfg.in_bands:
            raise ValueError(f"Input bands {c} != cfg.in_bands {self.cfg.in_bands}")

        x_chw = patch.permute(0, 3, 1, 2).contiguous()  # [B,C,P,P]
        feat = self.stem(x_chw)                         # [B,D,P,P]
        tokens, aux = self.token_learner(x_chw, feat)   # [B,K,D]
        tokens = self.mamba_blocks(tokens)              # [B,K,D]

        center_h = p_h // 2
        center_w = p_w // 2
        center_feature = feat[:, :, center_h, center_w]  # [B,D]
        token_summary = tokens.mean(dim=1)               # [B,D]
        fused = self.fusion(torch.cat([center_feature, token_summary], dim=-1))
        logits = self.classifier(fused)

        aux.update(
            {
                "tokens_after_mamba": tokens,
                "center_feature": center_feature,
                "token_summary": token_summary,
            }
        )
        return logits, aux


def build_lst_mamba(
    in_bands: int,
    num_classes: int,
    patch_size: int = 11,
    dim: int = 64,
    num_tokens: int = 8,
    rank: int = 1,
    mamba_depth: int = 1,
    d_state: int = 8,
    d_conv: int = 3,
    expand: int = 1,
    dropout: float = 0.05,
    bidirectional: bool = True,
) -> LSTMambaPatchClassifier:
    """Factory used by dataset-specific main scripts."""
    cfg = LSTMambaConfig(
        in_bands=in_bands,
        num_classes=num_classes,
        patch_size=patch_size,
        dim=dim,
        num_tokens=num_tokens,
        rank=rank,
        mamba_depth=mamba_depth,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return LSTMambaPatchClassifier(cfg)
