"""
12_evaluate_all_folds.py

Runs 11_evaluate_cv.py across all 5 CV folds for each trained model
configuration and prints a 5-fold summary (mean +/- std Dice and lesion-wise
F1 per class). Requires nnU-Net's --val fold outputs to already exist
(fold_N/validation/ under each trainer's results directory).
"""
import subprocess
import json
import numpy as np
from pathlib import Path

CONFIGS = {
    "primus_labelbased": {
        "results_base": Path("/workspace/nnUNet_results/Dataset001_BraTSMETS2026/nnUNet_PrimusV3S_Trainer__nnUNetPlans__3d_fullres"),
        "gt_dir": "/workspace/nnUNet_raw/Dataset001_BraTSMETS2026/labelsTr",
    },
    "primus_regionbased": {
        "results_base": Path("/workspace/nnUNet_results/Dataset002_BraTSMETS2026Regions/nnUNet_PrimusV3S_Trainer__nnUNetPlans__3d_fullres"),
        "gt_dir": "/workspace/nnUNet_raw/Dataset002_BraTSMETS2026Regions/labelsTr",
    },
    "resencl_labelbased": {
        "results_base": Path("/workspace/nnUNet_results/Dataset001_BraTSMETS2026/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"),
        "gt_dir": "/workspace/nnUNet_raw/Dataset001_BraTSMETS2026/labelsTr",
    },
    "plain_nnunet": {
        "results_base": Path("/workspace/nnUNet_results/Dataset001_BraTSMETS2026/nnUNetTrainer__nnUNetPlans__3d_fullres"),
        "gt_dir": "/workspace/nnUNet_raw/Dataset001_BraTSMETS2026/labelsTr",
    },
}

LABELS = ["NETC", "SNFH", "ET", "RC"]

for model_name, cfg in CONFIGS.items():
    print(f"\n{'='*50}\n{model_name}\n{'='*50}")
    all_fold_results = {}
    for fold in range(5):
        pred_dir = cfg["results_base"] / f"fold_{fold}" / "validation"
        if not pred_dir.exists():
            print(f"  fold {fold}: no validation dir found, skipping")
            continue
        out_json = f"cv_results_{model_name}_fold{fold}.json"
        subprocess.run(["python3", "11_evaluate_cv.py", str(pred_dir), cfg["gt_dir"], out_json], check=True)
        with open(out_json) as f:
            all_fold_results[f"fold_{fold}"] = json.load(f)

    print(f"\n--- {model_name}: 5-fold CV summary ---")
    for label in LABELS:
        dices, f1s = [], []
        for fold_data in all_fold_results.values():
            dices.extend(c[label]["dice"] for c in fold_data.values() if not np.isnan(c[label]["dice"]))
            f1s.extend(c[label]["f1"] for c in fold_data.values() if not np.isnan(c[label]["f1"]))
        if dices:
            print(f"{label}: Dice = {np.mean(dices):.4f} +/- {np.std(dices):.4f} (n={len(dices)})  |  F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (n={len(f1s)})")

    with open(f"cv_results_{model_name}_all_folds.json", "w") as f:
        json.dump(all_fold_results, f, indent=2)

print("\nDone.")
