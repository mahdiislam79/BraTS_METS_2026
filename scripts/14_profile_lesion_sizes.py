"""
14_profile_lesion_sizes.py

Profiles per-case lesion counts and volumes on the whole-tumor mask
(NETC+SNFH+ET, >=27mm^3 components) for a CV fold's held-out cases. Used to
identify clear small-lesion-dominant and large-lesion-dominant cases for
qualitative comparison figures (e.g. paper Figure 1) and for supporting the
paper's lesion-size framing of the "lesion-wise F1" metric name.
"""
import nibabel as nib
import numpy as np
from scipy import ndimage
from pathlib import Path
import json

LABELS_DIR = Path("/workspace/nnUNet_raw/Dataset001_BraTSMETS2026/labelsTr")
# use whichever fold's held-out case list you have predictions for from both models
FOLD_VAL_CASES = json.load(open("/workspace/nnUNet_preprocessed/Dataset001_BraTSMETS2026/splits_final.json"))[0]["val"]

MIN_VOLUME_MM3 = 27.0
profiles = {}

for case_id in FOLD_VAL_CASES:
    label_file = LABELS_DIR / f"{case_id}.nii.gz"
    if not label_file.exists():
        continue
    img = nib.load(label_file)
    data = img.get_fdata().astype(np.int32)
    zooms = img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(np.prod(zooms))

    # whole tumor mask (all non-background, non-RC tumor labels) for lesion sizing
    wt_mask = np.isin(data, [1, 2, 3])
    labeled, n = ndimage.label(wt_mask)
    lesion_volumes_mm3 = []
    for i in range(1, n + 1):
        vol = (labeled == i).sum() * voxel_vol_mm3
        if vol >= MIN_VOLUME_MM3:
            lesion_volumes_mm3.append(vol)

    if not lesion_volumes_mm3:
        continue

    profiles[case_id] = {
        "n_lesions": len(lesion_volumes_mm3),
        "total_volume_mm3": sum(lesion_volumes_mm3),
        "mean_lesion_volume_mm3": float(np.mean(lesion_volumes_mm3)),
        "max_lesion_volume_mm3": float(np.max(lesion_volumes_mm3)),
        "min_lesion_volume_mm3": float(np.min(lesion_volumes_mm3)),
    }

with open("case_lesion_profiles.json", "w") as f:
    json.dump(profiles, f, indent=2)

# rank by mean lesion size to find clear small- and large-lesion-dominant cases
sorted_by_mean = sorted(profiles.items(), key=lambda x: x[1]["mean_lesion_volume_mm3"])

print("=== Smallest mean-lesion-size cases (small-lesion candidates) ===")
for case_id, p in sorted_by_mean[:10]:
    print(f"{case_id}: n_lesions={p['n_lesions']}, mean_vol={p['mean_lesion_volume_mm3']:.0f}mm3, total={p['total_volume_mm3']:.0f}mm3")

print("\n=== Largest mean-lesion-size cases (large-lesion candidates) ===")
for case_id, p in sorted_by_mean[-10:]:
    print(f"{case_id}: n_lesions={p['n_lesions']}, mean_vol={p['mean_lesion_volume_mm3']:.0f}mm3, total={p['total_volume_mm3']:.0f}mm3")
