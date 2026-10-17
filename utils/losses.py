"""Loss functions for patch-based LST-Mamba.

The supervised objective is standard center-pixel cross entropy.  Two auxiliary
terms support the proposed method:

1. Spatial/spectral token diversity
   Different learned tokens should attend to different patch locations and
   spectral regions.  Orthogonality on normalized attention vectors discourages
   token collapse.

2. Center-pixel consistency
   HSI patch classification predicts the center pixel.  The Mamba token summary
   should provide contextual information without overwhelming the center pixel's
   own local feature.  A small MSE term keeps the token summary close to the
   center feature in the hidden space.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTTokenLoss(nn.Module):
    """CE + token diversity + center consistency for patch classification."""

    def __init__(
        self,
        lambda_spatial_div: float = 0.01,
        lambda_spectral_div: float = 0.01,
        lambda_center: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_spatial_div = float(lambda_spatial_div)
        self.lambda_spectral_div = float(lambda_spectral_div)
        self.lambda_center = float(lambda_center)

    @staticmethod
    def _zero_like_reference(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        for value in aux.values():
            if torch.is_tensor(value):
                return value.new_tensor(0.0)
        return torch.tensor(0.0)

    @staticmethod
    def _orthogonality_loss(attn: torch.Tensor) -> torch.Tensor:
        """Encourage K attention vectors to be diverse.

        Args:
            attn: [B, K, N], non-negative attention vectors.
        """
        attn = F.normalize(attn, p=2, dim=-1)
        gram = torch.matmul(attn, attn.transpose(1, 2))
        eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
        return (gram - eye).pow(2).mean()

    def forward(self, logits: torch.Tensor, aux: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Compute patch classification loss.

        Args:
            logits: [B, num_classes].
            aux: Dictionary returned by ``LSTMambaPatchClassifier``.
            target: [B] zero-based class ids.
        """
        target = target.long().view(-1)
        ce_loss = F.cross_entropy(logits, target)

        zero = self._zero_like_reference(aux)
        spatial_attn = aux.get("spatial_attn")
        spectral_attn = aux.get("spectral_attn")

        if spatial_attn is not None:
            spatial_div = self._orthogonality_loss(spatial_attn.flatten(2))
        else:
            spatial_div = zero
        if spectral_attn is not None:
            spectral_div = self._orthogonality_loss(spectral_attn)
        else:
            spectral_div = zero

        center_feature = aux.get("center_feature")
        token_summary = aux.get("token_summary")
        if center_feature is not None and token_summary is not None:
            center_loss = F.mse_loss(token_summary, center_feature)
        else:
            center_loss = zero

        return (
            ce_loss
            + self.lambda_spatial_div * spatial_div
            + self.lambda_spectral_div * spectral_div
            + self.lambda_center * center_loss
        )
