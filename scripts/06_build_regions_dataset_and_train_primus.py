"""
06_build_regions_dataset_and_train_primus.py

Builds Dataset002_BraTSMETS2026Regions (region-based, overlapping WT/TC/ET/RC
labels) by mirroring Dataset001's images and labels, writing a dataset.json
with overlapping region definitions. Then applies the corrected
regions_class_order (see paper Section 2.5 and script 10) BEFORE training,
since this dataset.json copy is what nnU-Net's preprocessing/training reads.

The actual preprocessing + training commands are shell commands -- run them
after this script via the printed instructions, or use
06b_train_primus_regionbased.sh directly if the dataset already exists.
"""
import os
import shutil
import json
from pathlib import Path

RAW = Path(os.environ.get("nnUNet_raw", "/workspace/nnUNet_raw"))
SRC = RAW / "Dataset001_BraTSMETS2026"
DST = RAW / "Dataset002_BraTSMETS2026Regions"

(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)

n = 0
for f in (SRC / "imagesTr").glob("*.nii.gz"):
    dst = DST / "imagesTr" / f.name
    if not dst.exists():
        shutil.copy2(f, dst)
    n += 1

for f in (SRC / "labelsTr").glob("*.nii.gz"):
    dst = DST / "labelsTr" / f.name
    if not dst.exists():
        shutil.copy2(f, dst)

# dataset.json with overlapping region definitions AND the corrected
# regions_class_order applied from the start (index 0 = whole_tumor -> label 2
# (SNFH, outermost), index 1 = tumor_core -> label 1 (NETC), index 2 =
# enhancing_tumor -> label 3 (ET), index 3 = resection_cavity -> label 4 (RC)).
# See paper Section 2.5 for why the naive ascending order [1,2,3,4] is wrong
# for this label scheme.
dataset_json = {
    "channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"},
    "labels": {
        "background": 0,
        "whole_tumor": [1, 2, 3],
        "tumor_core": [1, 3],
        "enhancing_tumor": [3],
        "resection_cavity": [4],
    },
    "regions_class_order": [2, 1, 3, 4],
    "numTraining": n,
    "file_ending": ".nii.gz",
}
with open(DST / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Dataset002 built: {n} cases mirrored, dataset.json written with corrected regions_class_order.")
print()
print("Next, run:")
print("  export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...")
print("  nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity -c 3d_fullres -np 8")
print("  (then run 06b_train_primus_regionbased.sh)")
