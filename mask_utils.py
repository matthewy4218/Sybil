"""
Mask preprocessing: raw segmentation mask → Sybil input-space tensor.

Mirrors Serie.get_volume() exactly so the mask stays pixel-aligned with the CT.
The output (200, 256, 256) float tensor is the format expected by MaskedSeriesDataset
and MaskAttentionLoss.

Supported mask file formats:
  .npy    numpy array (D, H, W), binary uint8/bool, slices in DICOM-sorted order
  .nii    NIfTI via torchio LabelMap (orientation handled automatically)
  .nii.gz same
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from sybil.datasets.utils import VOXEL_SPACING

# Must match Serie._load_args defaults
_TARGET_SIZE = (200, 256, 256)  # (T, H, W) in Sybil input space


def load_mask_file(mask_path: str) -> torch.Tensor:
    """
    Load a raw mask from disk.

    Returns a float Tensor of shape (D, H, W) with values in {0.0, 1.0}.
    For .npy the caller is responsible for ensuring slices are in the same
    order as the DICOMs (abdomen-first, clavicles-last).
    For NIfTI, torchio handles orientation automatically.
    """
    ext = mask_path.lower()
    if ext.endswith(".npy"):
        arr = np.load(mask_path).astype(np.float32)
        return torch.from_numpy(arr)
    elif ext.endswith(".nii.gz") or ext.endswith(".nii"):
        label_map = tio.LabelMap(mask_path)
        # torchio stores as (1, H, W, D); squeeze channel, put slices first
        data = label_map.data.squeeze(0)        # (H, W, D)
        data = data.permute(2, 0, 1).float()    # (D, H, W)
        return data
    else:
        raise ValueError(f"Unsupported mask format: {mask_path!r}. Use .npy or .nii(.gz).")


class MaskPreprocessor:
    """
    Resamples a segmentation mask to the (200, 256, 256) space that Sybil consumes,
    using the identical torchio pipeline as Serie.get_volume().

    Usage
    -----
    preprocessor = MaskPreprocessor(voxel_spacing=(0.7, 0.7, 2.5))
    mask_200 = preprocessor(raw_mask_tensor)   # (200, 256, 256)
    """

    def __init__(self, voxel_spacing: Tuple[float, float, float]):
        """
        Parameters
        ----------
        voxel_spacing : (row_spacing, col_spacing, slice_thickness)
            Original voxel spacing of the mask, in mm.
            This must match the CT's pixel/slice spacing (read from DICOM headers).
            Serie stores this as meta.voxel_spacing[:3].
        """
        # torchio affine diagonal: (row, col, slice, 1) — same convention as Serie
        spacing_tensor = torch.tensor(list(voxel_spacing) + [1], dtype=torch.float32)
        affine = torch.diag(spacing_tensor)

        self.affine = affine
        self.resample_transform = tio.transforms.Resample(target=VOXEL_SPACING)
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=(256, 256, 200),   # torchio uses (H, W, D)
            padding_mode=0,
        )

    def __call__(self, mask_raw: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mask_raw : Tensor (D, H, W), float, values in {0, 1}
            Raw mask in original DICOM voxel space.

        Returns
        -------
        Tensor (T=200, H=256, W=256), float, values in [0, 1]
            Mask aligned to Sybil's input space.
        """
        # torchio expects (C, H, W, D) for LabelMap
        # mask_raw is (D, H, W), so permute → (H, W, D), then add channel
        hwz = mask_raw.permute(1, 2, 0)          # (H, W, D)
        hwz = hwz.unsqueeze(0)                    # (1, H, W, D)

        label_map = tio.LabelMap(tensor=hwz, affine=self.affine)

        # Resample to VOXEL_SPACING using nearest-neighbour to keep binary values
        label_map = self._resample_nearest(label_map)

        # CropOrPad to (H=256, W=256, D=200) — same as Serie
        label_map = self.padding_transform(label_map)

        # Back to (T, H, W) convention
        result = label_map.data.squeeze(0)        # (H, W, D)
        result = result.permute(2, 0, 1).float()  # (D=200, H=256, W=256)
        return result

    def _resample_nearest(self, label_map: tio.LabelMap) -> tio.LabelMap:
        """Resample with nearest-neighbour interpolation to preserve binary values."""
        resample = tio.transforms.Resample(
            target=VOXEL_SPACING,
            image_interpolation="nearest",
        )
        return resample(label_map)


def mask_to_batch_fields(
    mask_200: torch.Tensor,
    max_followup: int = 6,
    eps: float = 1e-8,
) -> dict:
    """
    Convert a (T=200, H=256, W=256) mask in Sybil input space into the
    batch-dict fields expected by MaskAttentionLoss.

    The annotation is kept in input space (not downsampled here); the loss
    function performs the downsampling to feature-map resolution internally,
    matching how the original get_annotation_loss worked.

    Parameters
    ----------
    mask_200 : Tensor (200, 256, 256)
        Mask in Sybil input space.  Values in [0, 1].
    max_followup : int
        Max follow-up years (default 6, matches SybilNet).

    Returns
    -------
    dict with keys:
        image_annotations  : (1, T=200, H=256, W=256)   spatial mask
        annotation_areas   : (T=200,)                    area per slice
        has_annotation     : bool                        whether any voxel is annotated
        cancer_laterality  : (3,) float                 left/right/unknown — set by caller
    """
    # image_annotations: add channel dimension expected by loss
    image_annotations = mask_200.unsqueeze(0)   # (1, 200, 256, 256)

    # Per-slice total mask area (used for volume_attention supervision)
    annotation_areas = mask_200.sum(dim=(-2, -1))  # (200,)

    has_annotation = annotation_areas.sum() > eps

    # Laterality default: unknown (caller can override with actual laterality)
    cancer_laterality = torch.zeros(3)

    return {
        "image_annotations": image_annotations,
        "annotation_areas": annotation_areas,
        "has_annotation": has_annotation,
        "cancer_laterality": cancer_laterality,
    }
