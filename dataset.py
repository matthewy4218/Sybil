"""
Dataset for fine-tuning Sybil with segmentation mask supervision.

CSV format (one row per series):
    dicom_dir   — path to folder of DICOM files
    mask_path   — path to segmentation mask (.npy or .nii/.nii.gz), or empty
    label       — 1 if cancer, 0 otherwise
    censor_time — years to diagnosis (label=1) or last follow-up (label=0)

Mask files
----------
.npy    numpy array (D, H, W), dtype uint8/bool, slices in DICOM-sorted order
        (abdomen first, clavicles last — same order as DICOM sort by ImagePositionPatient)
.nii.gz NIfTI; torchio handles orientation automatically

When mask_path is empty or the file doesn't exist the sample is treated as
unannotated: has_annotation=False, image_annotations=zeros.
"""

import csv
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# These imports resolve relative to project root when sys.path includes it
from sybil.serie import Serie
from mask_supervision.mask_utils import MaskPreprocessor, load_mask_file, mask_to_batch_fields


class MaskedSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapping (DICOM series, segmentation mask) pairs.

    Each __getitem__ returns a dict suitable as input to the fine-tuning loop.

    Parameters
    ----------
    csv_path : str
        Path to the training CSV (columns: dicom_dir, mask_path, label, censor_time).
    max_followup : int
        Yearly bins for the survival label (default 6, matching Sybil).
    """

    def __init__(self, csv_path: str, max_followup: int = 6):
        self.max_followup = max_followup
        self.rows = self._load_csv(csv_path)

    @staticmethod
    def _load_csv(path: str) -> List[dict]:
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        dicom_dir = row["dicom_dir"].strip()
        mask_path = row.get("mask_path", "").strip()
        label = int(row["label"])
        censor_time = float(row["censor_time"])

        # ----------------------------------------------------------------
        # Load CT volume via Serie (identical to inference path)
        # ----------------------------------------------------------------
        dicom_files = sorted([
            os.path.join(dicom_dir, f)
            for f in os.listdir(dicom_dir)
            if os.path.isfile(os.path.join(dicom_dir, f)) and not f.startswith(".")
        ])
        serie = Serie(
            dicom_files,
            label=label,
            censor_time=int(censor_time),   # Serie expects int years
            split="train",
        )
        volume = serie.get_volume().squeeze(0)  # (C=3, T=200, H=256, W=256)

        # Survival label tensors
        label_obj = serie.get_label(self.max_followup)
        y_seq = torch.tensor(label_obj.y_seq, dtype=torch.float32)    # (max_followup,)
        y_mask = torch.tensor(label_obj.y_mask, dtype=torch.float32)  # (max_followup,)

        # ----------------------------------------------------------------
        # Load and preprocess segmentation mask (optional)
        # ----------------------------------------------------------------
        mask_fields = self._load_mask(mask_path, serie)

        return {
            "volume": volume,                                      # (3, 200, 256, 256)
            "y": torch.tensor(label_obj.y, dtype=torch.long),     # scalar
            "y_seq": y_seq,                                        # (6,)
            "y_mask": y_mask,                                      # (6,)
            "time_at_event": torch.tensor(label_obj.censor_time, dtype=torch.long),
            **mask_fields,
        }

    def _load_mask(self, mask_path: str, serie: Serie) -> dict:
        """Load mask and preprocess; return zero mask if not available."""
        has_mask = mask_path and os.path.exists(mask_path)

        if not has_mask:
            # Return zero mask — has_annotation=False suppresses the attn loss
            return {
                "image_annotations": torch.zeros(1, 200, 256, 256),
                "annotation_areas": torch.zeros(200),
                "has_annotation": torch.tensor(False),
                "cancer_laterality": torch.zeros(3),
            }

        # Get voxel spacing from the DICOM series
        ps = serie._meta.pixel_spacing       # [row_spacing, col_spacing]
        thickness = serie._meta.thickness    # slice thickness in mm
        voxel_spacing = (ps[0], ps[1], thickness)

        preprocessor = MaskPreprocessor(voxel_spacing)
        raw_mask = load_mask_file(mask_path)     # (D, H, W)
        mask_200 = preprocessor(raw_mask)         # (200, 256, 256)

        fields = mask_to_batch_fields(mask_200, self.max_followup)
        # Convert scalar bool to tensor for collation
        fields["has_annotation"] = torch.tensor(fields["has_annotation"])
        return fields


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate: stack tensors, handle bool has_annotation.
    Pass this as collate_fn to DataLoader.
    """
    keys = batch[0].keys()
    out = {}
    for key in keys:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals, dim=0)
    return out
