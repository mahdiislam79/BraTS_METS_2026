"""
10_postprocess_predictions.py

Reusable postprocessing for any model's predicted label maps: per-class
connected-component volume-threshold filtering (>=27 mm^3, matching the
challenge's own scoring threshold) followed by hole-filling on the
size-filtered mask.

Deliberately does NOT use scipy.ndimage.binary_opening -- morphological
opening's erosion step was found to eliminate connected components near the
27 mm^3 scoring threshold before dilation can restore them, collapsing
lesion-wise F1 to near-zero (0.01-0.02) despite reasonable DSC. Removing
opening and replacing it with this volume-threshold + hole-fill approach
recovered lesion-wise F1 from ~0.01 to ~0.33-0.41 across ET/TC/WT
(see paper Section 2.6).

Usage:
    python3 10_postprocess_predictions.py <pred_dir> <out_dir>
"""
import sys
import nibabel as nib
import numpy as np
from scipy import ndimage
from pathlib import Path

MIN_VOLUME_MM3 = 27.0
LABELS = [1, 2, 3, 4]  # NETC, SNFH, ET, RC


def postprocess_file(pred_file, out_dir):
    img = nib.load(pred_file)
    data = img.get_fdata().astype(np.int32)
    zooms = img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    cleaned = np.zeros_like(data)
    for label in LABELS:
        mask = (data == label)
        if not mask.any():
            continue

        # 1. Connected-component size filtering (NO binary_opening -- see module docstring)
        labeled, n_components = ndimage.label(mask)
        kept_mask = np.zeros_like(mask)
        for i in range(1, n_components + 1):
            comp = labeled == i
            if comp.sum() * voxel_vol_mm3 >= MIN_VOLUME_MM3:
                kept_mask |= comp

        # 2. Hole-filling on the size-filtered mask (per-class, safe -- fills
        #    small internal gaps within a real lesion without merging
        #    separate lesions, unlike closing/opening on the whole mask).
        filled_mask = ndimage.binary_fill_holes(kept_mask)
        cleaned[filled_mask] = label

    nib.save(nib.Nifti1Image(cleaned, img.affine, img.header), out_dir / pred_file.name)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 10_postprocess_predictions.py <pred_dir> <out_dir>")
        sys.exit(1)

    pred_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    for pred_file in sorted(pred_dir.glob("*.nii.gz")):
        postprocess_file(pred_file, out_dir)
        n_processed += 1

    print(f"Postprocessed {n_processed} files -> {out_dir}")
