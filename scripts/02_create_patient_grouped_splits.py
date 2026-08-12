"""
02_create_patient_grouped_splits.py

Builds patient-grouped 5-fold cross-validation splits (GroupKFold) to prevent
the temporal leakage that would occur with naive random/modulo splitting on
the longitudinal UCSD subset, where each patient contributes multiple imaging
timepoints encoded in the case identifier (see paper Section 2.2).

Writes nnU-Net's splits_final.json so this exact fold assignment is reused
identically across every model variant trained on Dataset001.
"""
import os
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold

labels_tr = Path(os.environ.get("nnUNet_preprocessed", "/workspace/nnUNet_preprocessed")).parent / "nnUNet_raw" / "Dataset001_BraTSMETS2026" / "labelsTr"
# fallback if nnUNet_raw is set directly
if not labels_tr.exists():
    labels_tr = Path(os.environ.get("nnUNet_raw", "/workspace/nnUNet_raw")) / "Dataset001_BraTSMETS2026" / "labelsTr"

case_ids = sorted(f.stem.replace(".nii", "") for f in labels_tr.glob("*.nii.gz"))
# Patient ID = case ID with the trailing timepoint suffix stripped
# (e.g. "BraTS-MET-00001-002" -> "BraTS-MET-00001")
patient_ids = [cid.rsplit("-", 1)[0] for cid in case_ids]

gkf = GroupKFold(n_splits=5)
splits = []
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(case_ids, groups=patient_ids)):
    splits.append({
        "train": [case_ids[i] for i in train_idx],
        "val": [case_ids[i] for i in val_idx]
    })

out_dir = Path(os.environ.get("nnUNet_preprocessed", "/workspace/nnUNet_preprocessed")) / "Dataset001_BraTSMETS2026"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "splits_final.json", "w") as f:
    json.dump(splits, f, indent=2)

print(f"splits_final.json written, {len(case_ids)} cases across 5 folds")
print("NOTE: copy this same splits_final.json into the Dataset002 preprocessed")
print("      folder before training region-based models, so all architectures")
print("      share identical fold assignments for fair cross-model comparison.")
