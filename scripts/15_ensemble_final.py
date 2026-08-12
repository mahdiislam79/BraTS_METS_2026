"""
15_ensemble_final.py

Ensemble of verified, bug-free label-based models only:
  - Primus (label-based, Dataset001) -- strongest large/medium lesion DSC/NSD
  - ResEncL (label-based, Dataset001) -- strongest small-instance detection F1

Deliberately EXCLUDES both region-based variants:
  - ResEncL region-based: regions_class_order fix applied but not re-verified at scale
  - Primus region-based: confirmed 23.7% RC false-positive rate on RC-negative
    cases, plus an unresolved Synapse scoring anomaly on an earlier submission

Run this in TWO modes:
  MODE = "cv"    -> verify against ground truth (fold_0/validation, has labels)
                    BEFORE submitting
  MODE = "blind" -> generate the real submission on imagesTs (blind validation set)

Weights: near-equal, slight edge to ResEncL (1.15 vs 1.0) -- Primus leads on
DSC/NSD for large/medium lesions, but ResEncL's small-instance F1 (0.35-0.45
vs Primus's 0.04-0.05) is this project's single biggest weakness relative to
competing submissions, and small-instance F1 feeds directly into the official
lesion-wise F1 ranking metric -- worth protecting rather than letting Primus
dominate it out. (See paper Discussion: probability-averaging did NOT fully
preserve this protection in practice, due to the fixed 0.5 threshold
suppressing ResEncL's minority-correct predictions when averaged with
Primus's near-zero probability on lesions it misses.)
"""
import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path

MODE = "blind"  # "cv" to verify against ground truth first, "blind" for the real submission

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_VOLUME_MM3 = 27.0
THRESHOLD = 0.5

MODELS = {
    "primus_labelbased": {
        "weight": 1.0,
        "cv_dir": "/workspace/nnUNet_results/Dataset001_BraTSMETS2026/nnUNet_PrimusV3S_Trainer__nnUNetPlans__3d_fullres/fold_0/validation",
        "blind_npz_dir": "/workspace/predictions/predictions_primus_validation",
    },
    "resencl_labelbased": {
        "weight": 1.15,
        "cv_dir": "/workspace/nnUNet_results/Dataset001_BraTSMETS2026/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation",
        "blind_npz_dir": "/workspace/predictions/predictions_resencl_labelbased_validation",
    },
}

GT_DIR_CV = "/workspace/nnUNet_preprocessed/Dataset001_BraTSMETS2026/gt_segmentations"
REF_DIR_BLIND = "/workspace/nnUNet_raw/Dataset001_BraTSMETS2026/imagesTs"

OUT_DIR = Path(f"/workspace/ensemble_final_{MODE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Softmax (5-channel: bg, NETC, SNFH, ET, RC) -> region-probability (4-channel: WT,TC,ET,RC)
# ---------------------------------------------------------------------------

def softmax_to_regions(prob):
    # nnU-Net's saved npz probabilities are in (C, Z, Y, X) order --
    # transpose spatial axes to (X, Y, Z) to match nib.get_fdata()'s orientation,
    # which everything else in this script (reference affine, postprocessing) uses
    prob = np.transpose(prob, (0, 3, 2, 1))
    wt = prob[1] + prob[2] + prob[3]
    tc = prob[1] + prob[3]
    et = prob[3]
    rc = prob[4]
    return np.stack([wt, tc, et, rc])


def postprocess_label_map(label_map, voxel_vol_mm3):
    cleaned = np.zeros_like(label_map)
    for label in [1, 2, 3, 4]:
        mask = (label_map == label)
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        kept = np.zeros_like(mask)
        for i in range(1, n + 1):
            comp = labeled == i
            if comp.sum() * voxel_vol_mm3 >= MIN_VOLUME_MM3:
                kept |= comp
        if kept.any():
            cleaned[ndimage.binary_fill_holes(kept)] = label
    return cleaned


def reconstruct_from_regions(wt, tc, et, rc, threshold=THRESHOLD):
    label_map = np.zeros(wt.shape, dtype=np.int32)
    label_map[wt > threshold] = 2  # SNFH (outer, applied first)
    label_map[tc > threshold] = 1  # NETC (overwrites WT region)
    label_map[et > threshold] = 3  # ET (overwrites TC region)
    label_map[rc > threshold] = 4  # RC (independent, applied last)
    return label_map


# ---------------------------------------------------------------------------
# CV mode: verify against ground truth using nnU-Net's own npz probability
# exports from --val (requires each model's fold_0/validation/*.npz to exist)
# ---------------------------------------------------------------------------

def run_cv():
    case_ids = sorted(f.stem for f in Path(MODELS["primus_labelbased"]["cv_dir"]).glob("*.npz"))
    print(f"Ensembling {len(case_ids)} CV cases (fold 0, ground truth available)")

    for case_id in case_ids:
        combined, total_weight = None, 0.0
        ref_img = None

        for model_name, cfg in MODELS.items():
            npz_path = Path(cfg["cv_dir"]) / f"{case_id}.npz"
            nii_path = Path(cfg["cv_dir"]) / f"{case_id}.nii.gz"
            if not npz_path.exists():
                print(f"  WARNING: {model_name} missing npz for {case_id}, skipping this model for this case")
                continue

            data = np.load(npz_path)
            prob = data["probabilities"]
            region_probs = softmax_to_regions(prob)  # all models here are label-based/softmax

            w = cfg["weight"]
            combined = region_probs * w if combined is None else combined + region_probs * w
            total_weight += w

            if ref_img is None and nii_path.exists():
                ref_img = nib.load(nii_path)

        if combined is None or ref_img is None:
            print(f"  SKIP {case_id}: insufficient data")
            continue

        combined /= total_weight
        wt, tc, et, rc = combined
        label_map = reconstruct_from_regions(wt, tc, et, rc)

        voxel_vol_mm3 = float(np.prod(ref_img.header.get_zooms()[:3]))
        cleaned = postprocess_label_map(label_map, voxel_vol_mm3)

        nib.save(nib.Nifti1Image(cleaned.astype(np.uint8), ref_img.affine, ref_img.header),
                  OUT_DIR / f"{case_id}.nii.gz")

    n_out = len(list(OUT_DIR.glob("*.nii.gz")))
    print(f"Done. {n_out} CV predictions written to {OUT_DIR}")
    print(f"\nNext: run 11_evaluate_cv.py against {OUT_DIR} and {GT_DIR_CV} to get a real Dice/NSD/F1 score")
    print(f"  python3 11_evaluate_cv.py {OUT_DIR} {GT_DIR_CV} ensemble_cv_results.json")


# ---------------------------------------------------------------------------
# Blind mode: generate the actual submission on imagesTs (no ground truth available)
# ---------------------------------------------------------------------------

def run_blind():
    case_ids = sorted(f.stem for f in Path(MODELS["primus_labelbased"]["blind_npz_dir"]).glob("*.npz"))
    print(f"Ensembling {len(case_ids)} blind validation-set cases")

    for case_id in case_ids:
        combined, total_weight = None, 0.0

        for model_name, cfg in MODELS.items():
            npz_path = Path(cfg["blind_npz_dir"]) / f"{case_id}.npz"
            if not npz_path.exists():
                print(f"  WARNING: {model_name} missing npz for {case_id}, skipping this model for this case")
                continue
            data = np.load(npz_path)
            prob = data["probabilities"]
            region_probs = softmax_to_regions(prob)
            w = cfg["weight"]
            combined = region_probs * w if combined is None else combined + region_probs * w
            total_weight += w

        if combined is None:
            print(f"  SKIP {case_id}: no model predictions found")
            continue

        combined /= total_weight
        wt, tc, et, rc = combined
        label_map = reconstruct_from_regions(wt, tc, et, rc)

        ref_path = Path(REF_DIR_BLIND) / f"{case_id}_0000.nii.gz"
        if not ref_path.exists():
            print(f"  SKIP {case_id}: no reference image for affine/header")
            continue
        ref_img = nib.load(ref_path)
        voxel_vol_mm3 = float(np.prod(ref_img.header.get_zooms()[:3]))
        cleaned = postprocess_label_map(label_map, voxel_vol_mm3)

        nib.save(nib.Nifti1Image(cleaned.astype(np.uint8), ref_img.affine, ref_img.header),
                  OUT_DIR / f"{case_id}.nii.gz")

    n_out = len(list(OUT_DIR.glob("*.nii.gz")))
    print(f"Done. {n_out} blind predictions written to {OUT_DIR}")
    print(f"\nNext: zip and submit --")
    print(f"  cd {OUT_DIR} && zip -j /workspace/submission_ensemble_final.zip *.nii.gz")


if __name__ == "__main__":
    if MODE == "cv":
        run_cv()
    elif MODE == "blind":
        run_blind()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
