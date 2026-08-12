"""
09_reconstruct_regions_from_probs.py

Post-hoc correction for predictions already generated under the buggy
ascending regions_class_order [1,2,3,4]. Reconstructs discrete label maps
directly from the saved region-probability .npz files using the corrected
order -- outer region first (SNFH), overwritten by the inner tumor-core
region (NETC), then ET, then the independent RC class -- WITHOUT needing to
retrain or re-run inference (see paper Section 2.5).

Region channel order in the saved .npz (index -> nnU-Net region definition):
  0 = whole_tumor,  1 = tumor_core,  2 = enhancing_tumor,  3 = resection_cavity
Corrected label assignment: WT->2 (SNFH), TC->1 (NETC), ET->3, RC->4,
applied in this sequence so later assignments correctly overwrite earlier ones.
"""
import numpy as np
import nibabel as nib
from pathlib import Path

NPZ_DIR = Path("/workspace/predictions_primus_regions_validation")
REF_NII_DIR = Path("/workspace/predictions_primus_regions_validation")  # original (buggy) exports, for affine/header reference
OUT_DIR = Path("/workspace/predictions_primus_regions_validation_FIXED")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = [
    (0, 2),  # whole_tumor probs -> label 2 (SNFH), applied first (outer shell)
    (1, 1),  # tumor_core probs  -> label 1 (NETC), overwrites WT region where TC positive
    (2, 3),  # enhancing_tumor   -> label 3 (ET), overwrites TC region where ET positive
    (3, 4),  # resection_cavity  -> label 4 (RC), independent, applied last
]

THRESHOLD = 0.5

n_processed = 0
for npz_file in sorted(NPZ_DIR.glob("*.npz")):
    case_id = npz_file.stem
    ref_nii_path = REF_NII_DIR / f"{case_id}.nii.gz"
    if not ref_nii_path.exists():
        print(f"SKIP {case_id}: no reference nii for affine/header")
        continue

    ref_img = nib.load(ref_nii_path)

    data = np.load(npz_file)
    key = list(data.keys())[0]
    prob = data[key]  # shape: (4, X, Y, Z)

    label_map = np.zeros(prob.shape[1:], dtype=np.int32)
    for channel_idx, label_value in REGION_ORDER:
        mask = prob[channel_idx] > THRESHOLD
        label_map[mask] = label_value

    nib.save(nib.Nifti1Image(label_map, ref_img.affine, ref_img.header), OUT_DIR / f"{case_id}.nii.gz")
    n_processed += 1

print(f"Reconstructed {n_processed} cases -> {OUT_DIR}")
