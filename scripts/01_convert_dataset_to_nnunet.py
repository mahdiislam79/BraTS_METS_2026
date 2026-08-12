"""
01_convert_dataset_to_nnunet.py

Converts the raw BraTS-METS 2025 Lighthouse training data (per-case folders,
each containing t1n/t1c/t2w/t2f + seg NIfTI files) into nnU-Net v2 raw format
for Dataset001_BraTSMETS2026 (label-based, mutually-exclusive per-voxel classes).

Applies the official corrected-labels overlay where available, and excludes
the single case whose out-of-scheme labels were not covered by that overlay
(129 voxels affected; see paper Section 2.1).

Requires: nnUNet_raw env var set (or edit NNUNET_RAW below).
"""
import os
import shutil
import json
from pathlib import Path

SRC_ROOT = Path("/workspace/Dataset/Training")
CORRECTED_LABELS_DIR = Path("/workspace/Dataset/corrected-labels")
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", "/workspace/nnUNet_raw"))
DATASET_NAME = "Dataset001_BraTSMETS2026"

MODALITY_MAP = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}
# Case with out-of-scheme labels (129 voxels) not covered by the official
# corrected-labels overlay -- excluded from training (see paper Section 2.1).
EXCLUDE_CASES = {"BraTS-MET-01094-002"}

out_dir = NNUNET_RAW / DATASET_NAME
images_tr = out_dir / "imagesTr"
labels_tr = out_dir / "labelsTr"
images_tr.mkdir(parents=True, exist_ok=True)
labels_tr.mkdir(parents=True, exist_ok=True)

corrected_case_ids = {
    f.name.replace("-seg.nii.gz", "") for f in CORRECTED_LABELS_DIR.glob("*-seg.nii.gz")
} if CORRECTED_LABELS_DIR.exists() else set()

seg_files = sorted(SRC_ROOT.rglob("*-seg.nii.gz"))
print(f"Found {len(seg_files)} case folders (recursive search)")

n_ok, n_skipped, n_excluded, n_corrected = 0, 0, 0, 0
for seg_file in seg_files:
    case_dir = seg_file.parent
    case_id = seg_file.name.replace("-seg.nii.gz", "")

    if case_id in EXCLUDE_CASES:
        n_excluded += 1
        continue

    mod_files = {mod: case_dir / f"{case_id}-{mod}.nii.gz" for mod in MODALITY_MAP}
    missing = [str(p) for p in mod_files.values() if not p.exists()]
    if missing:
        # Conservative complete-modality inclusion criterion (T2w is
        # non-mandatory from 2025 onward and absent for several cases).
        print(f"SKIPPING {case_id}, missing: {missing}")
        n_skipped += 1
        continue

    for mod, channel_idx in MODALITY_MAP.items():
        dst = images_tr / f"{case_id}_{channel_idx}.nii.gz"
        if not dst.exists():
            shutil.copy2(mod_files[mod], dst)

    if case_id in corrected_case_ids:
        label_src = CORRECTED_LABELS_DIR / f"{case_id}-seg.nii.gz"
        n_corrected += 1
    else:
        label_src = seg_file

    dst_label = labels_tr / f"{case_id}.nii.gz"
    if not dst_label.exists():
        shutil.copy2(label_src, dst_label)

    n_ok += 1

print(f"\nConverted: {n_ok}, Skipped: {n_skipped}, Excluded: {n_excluded}, Corrected: {n_corrected}")

dataset_json = {
    "channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"},
    "labels": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4},
    "numTraining": n_ok,
    "file_ending": ".nii.gz"
}
with open(out_dir / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"dataset.json written to {out_dir / 'dataset.json'}")
