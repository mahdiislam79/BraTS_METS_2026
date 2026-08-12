"""
11_evaluate_cv.py

Internal cross-validation evaluator: computes lesion-wise DSC, NSD (tolerance
2.0mm), and lesion-wise F1 (27mm^3 volume floor, greedy any-voxel-overlap
matching, 6-connectivity via scipy.ndimage.label default) against ground
truth. Used for internal CV model selection and ablation checks prior to
submission.

NOTE: the paper's reported DSC/NSD/F1 values (Tables 1-2) are the OFFICIAL
BraTS-METS validation-server scores, not this script's output -- see
results/ for the official per-submission CSVs. This script is provided for
reproducibility of the internal fold-selection process only.

Usage:
    python3 11_evaluate_cv.py <pred_dir> <gt_dir> [output_json]
"""
import nibabel as nib
import numpy as np
from scipy import ndimage
from pathlib import Path
import json
import sys

try:
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
    HAVE_SURFACE_DISTANCE = True
except ImportError:
    HAVE_SURFACE_DISTANCE = False
    print("WARNING: install 'surface-distance' package for NSD (pip install surface-distance)")

MIN_VOLUME_MM3 = 27.0
LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
NSD_TOLERANCE_MM = 2.0


def get_components_above_threshold(mask, voxel_vol_mm3, min_vol=MIN_VOLUME_MM3):
    labeled, n = ndimage.label(mask)
    comps = []
    for i in range(1, n + 1):
        comp_mask = labeled == i
        vol = comp_mask.sum() * voxel_vol_mm3
        if vol >= min_vol:
            comps.append(comp_mask)
    return comps


def dice(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return 2 * inter / denom if denom > 0 else np.nan


def lesion_wise_f1(pred_mask, gt_mask, voxel_vol_mm3, overlap_thresh=1e-6):
    gt_comps = get_components_above_threshold(gt_mask, voxel_vol_mm3)
    pred_comps = get_components_above_threshold(pred_mask, voxel_vol_mm3)

    matched_gt = set()
    matched_pred = set()
    for pi, pc in enumerate(pred_comps):
        for gi, gc in enumerate(gt_comps):
            if gi in matched_gt:
                continue
            if np.logical_and(pc, gc).sum() > overlap_thresh:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    fp = len(pred_comps) - len(matched_pred)
    fn = len(gt_comps) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and (precision + recall) > 0) else np.nan
    return f1, precision, recall, tp, fp, fn


def evaluate_case(pred_path, gt_path):
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)
    pred_data = pred_img.get_fdata().astype(np.int32)
    gt_data = gt_img.get_fdata().astype(np.int32)
    zooms = pred_img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    results = {}
    for label, name in LABELS.items():
        pred_mask = pred_data == label
        gt_mask = gt_data == label

        d = dice(pred_mask, gt_mask)
        f1, prec, rec, tp, fp, fn = lesion_wise_f1(pred_mask, gt_mask, voxel_vol_mm3)

        nsd = np.nan
        if HAVE_SURFACE_DISTANCE and pred_mask.any() and gt_mask.any():
            sd = compute_surface_distances(gt_mask, pred_mask, spacing_mm=zooms)
            nsd = compute_surface_dice_at_tolerance(sd, NSD_TOLERANCE_MM)

        results[name] = {"dice": d, "nsd": nsd, "f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 11_evaluate_cv.py <pred_dir> <gt_dir> [output_json]")
        sys.exit(1)

    pred_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    out_json = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("cv_eval_results.json")

    all_results = {}
    for pred_file in sorted(pred_dir.glob("*.nii.gz")):
        case_id = pred_file.name.replace(".nii.gz", "")
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            continue
        all_results[case_id] = evaluate_case(pred_file, gt_file)

    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    for name in LABELS.values():
        dices = [r[name]["dice"] for r in all_results.values() if not np.isnan(r[name]["dice"])]
        f1s = [r[name]["f1"] for r in all_results.values() if not np.isnan(r[name]["f1"])]
        print(f"{name}: mean Dice={np.mean(dices):.4f} (n={len(dices)}), mean lesion F1={np.mean(f1s):.4f} (n={len(f1s)})")
