"""
Mask-supervised attention loss for Sybil fine-tuning.

Replaces the bounding-box target in the original get_annotation_loss
(sybil/utils/losses.py) with a pixel-accurate segmentation mask target.

Key differences from the original:
- Uses max-pool (not area interpolation) to downsample the mask to feature-map
  resolution, so small nodules are not averaged away.
- Skips the laterality loss (cancer_laterality) — it is not needed when the
  mask already encodes the cancer-side voxels explicitly.
- Supports a single lambda hyperparameter instead of three separate lambdas.

Loss breakdown:
    image_attn_loss  — KL(mask_image_target || image_attention_1)
                       Supervises spatial attention per slice.
    volume_attn_loss — KL(mask_volume_target || volume_attention_1)
                       Supervises which slices attend to cancer.
    total_attn_loss  = lambda_mask * (image_attn_loss + volume_attn_loss)
"""

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature-map dimensions — verified empirically by printing image_encoder(x).shape
# on the real 200-slice, 256x256 input.
_T_PRIME = 25
_H_PRIME = 16
_W_PRIME = 16
_POOL_KERNEL = (
    200 // _T_PRIME,   # 8
    256 // _H_PRIME,   # 16
    256 // _W_PRIME,   # 16
)


def _downsample_mask_to_feature_space(
    image_annotations: torch.Tensor,
    batch_has_annotation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Max-pool the mask from input space (B, 1, 200, 256, 256) to feature-map
    space (B, T'=25, H'=16, W'=16), then derive per-slice and per-spatial targets.

    Using max-pool (not area/trilinear) ensures small nodules survive.

    Returns
    -------
    mask_feat      : (B, T', H', W')   binary/soft mask at feature resolution
    image_target   : (B, T', H'*W')    per-slice spatial probability distribution
    volume_target  : (B, T')           per-volume slice probability distribution
    valid_slice    : (B, T')           which slices have any mask content
    """
    B = image_annotations.shape[0]

    # annotations come in as (B, 1, T, H, W) or (B, T, H, W)
    if image_annotations.dim() == 4:
        image_annotations = image_annotations.unsqueeze(1)  # (B, 1, T, H, W)

    # max-pool to feature-map resolution
    mask_feat = F.max_pool3d(
        image_annotations.float(),
        kernel_size=_POOL_KERNEL,
        stride=_POOL_KERNEL,
    )  # (B, 1, T', H', W')
    mask_feat = mask_feat.squeeze(1)  # (B, T', H', W')

    # Zero out samples that have no annotation
    mask_feat = mask_feat * batch_has_annotation.float()[:, None, None, None]

    # --- image_attention target: per-slice spatial distribution (B, T', H'*W') ---
    flat = mask_feat.view(B, _T_PRIME, -1)          # (B, T', H'*W')
    slice_mass = flat.sum(dim=-1, keepdim=True)      # (B, T', 1)
    valid_slice = slice_mass.squeeze(-1) > 1e-8      # (B, T') bool

    image_target = torch.zeros_like(flat)
    # Only normalise slices that actually have mask content
    image_target[valid_slice] = flat[valid_slice] / slice_mass.expand_as(flat)[valid_slice]

    # --- volume_attention target: per-volume slice distribution (B, T') ---
    volume_mass = slice_mass.squeeze(-1)             # (B, T')
    total_mass = volume_mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    volume_target = volume_mass / total_mass         # (B, T'), sums to 1

    return mask_feat, image_target, volume_target, valid_slice


class MaskAttentionLoss(nn.Module):
    """
    KL-divergence loss between Sybil's attention maps and mask-derived targets.

    Both image_attention_1 and volume_attention_1 are already in log-space
    (output of LogSoftmax in pooling_layer.py), so F.kl_div is applied directly.

    Parameters
    ----------
    lambda_image  : weight on the spatial (image) attention loss
    lambda_volume : weight on the slice (volume) attention loss
    """

    def __init__(self, lambda_image: float = 0.1, lambda_volume: float = 0.1):
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_volume = lambda_volume

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Parameters
        ----------
        model_output : dict from SybilNet.forward()
            Must contain 'image_attention_1' (B, T', H'*W') and
            'volume_attention_1' (B, T').  Both are log-softmax outputs.
        batch : dict from MaskedSeriesDataset
            Must contain:
              'image_annotations' (B, [1,] T=200, H=256, W=256)
              'has_annotation'    (B,) bool/float

        Returns
        -------
        total_loss : scalar Tensor
        logging    : OrderedDict with component losses for monitoring
        """
        logging = OrderedDict()

        log_image_attn = model_output["image_attention_1"]   # (B, T', H'*W')
        log_volume_attn = model_output["volume_attention_1"] # (B, T')

        has_annotation = batch["has_annotation"].bool()      # (B,)

        # Skip the entire loss if no sample in this batch has a mask
        if not has_annotation.any():
            zero = log_image_attn.new_zeros(())
            logging["image_attn_loss"] = zero.detach()
            logging["volume_attn_loss"] = zero.detach()
            logging["total_attn_loss"] = zero.detach()
            return zero, logging

        _, image_target, volume_target, valid_slice = _downsample_mask_to_feature_space(
            batch["image_annotations"], has_annotation
        )

        # Move targets to same device as model outputs
        image_target = image_target.to(log_image_attn.device)
        volume_target = volume_target.to(log_volume_attn.device)
        valid_slice = valid_slice.to(log_image_attn.device)

        # ------------------------------------------------------------------
        # Image attention loss
        # KL(target || predicted) where log_image_attn = log p(predicted)
        # F.kl_div(log_p, q) computes q * (log q - log_p)
        # ------------------------------------------------------------------
        image_kl = F.kl_div(
            log_image_attn,    # (B, T', H'*W') — log probs
            image_target,      # (B, T', H'*W') — probs
            reduction="none",
            log_target=False,
        )  # (B, T', H'*W')

        # Sum over spatial dim, mask out slices with no content
        image_kl_per_slice = image_kl.sum(dim=-1)                              # (B, T')
        image_kl_per_slice = image_kl_per_slice * valid_slice.float()

        n_valid = valid_slice.float().sum(dim=-1).clamp(min=1.0)               # (B,)
        image_attn_loss = (image_kl_per_slice.sum(dim=-1) / n_valid).mean()

        # ------------------------------------------------------------------
        # Volume attention loss
        # Only average over samples that actually have an annotation
        # ------------------------------------------------------------------
        volume_kl = F.kl_div(
            log_volume_attn,   # (B, T') — log probs
            volume_target,     # (B, T') — probs
            reduction="none",
            log_target=False,
        )  # (B, T')

        volume_kl_per_sample = volume_kl.sum(dim=-1)                           # (B,)
        n_annotated = has_annotation.float().sum().clamp(min=1.0)
        volume_attn_loss = (
            volume_kl_per_sample * has_annotation.float()
        ).sum() / n_annotated

        # ------------------------------------------------------------------
        # Combined
        # ------------------------------------------------------------------
        total_attn_loss = (
            self.lambda_image * image_attn_loss
            + self.lambda_volume * volume_attn_loss
        )

        logging["image_attn_loss"] = image_attn_loss.detach()
        logging["volume_attn_loss"] = volume_attn_loss.detach()
        logging["total_attn_loss"] = total_attn_loss.detach()

        return total_attn_loss, logging
