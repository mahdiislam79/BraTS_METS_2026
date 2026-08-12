# Run order

Extracted directly from `Primus.ipynb` and `Ensemble.ipynb`. Filenames are numbered in
pipeline order; place `configs/*.json` under `configs/` and `scripts/*` under `scripts/`
in the GitHub repo structure discussed earlier in this conversation.

1. **`scripts/01_convert_dataset_to_nnunet.py`** — raw BraTS-METS data -> Dataset001 (label-based) nnU-Net raw format
2. **`scripts/02_create_patient_grouped_splits.py`** — patient-grouped 5-fold `splits_final.json`
3. **`scripts/03_train_plain_nnunet.sh`** — preprocess + train plain nnU-Net baseline
4. **`scripts/04_train_resencl.sh`** — plan/preprocess + train ResEncL
5. **`scripts/05_train_primus_labelbased.sh`** — preprocess + train Primus (label-based)
6. **`scripts/06_build_regions_dataset_and_train_primus.py`** — build Dataset002 (region-based) with corrected `regions_class_order` baked in from the start
   **`scripts/06b_train_primus_regionbased.sh`** — train Primus on Dataset002 (run `nnUNetv2_plan_and_preprocess -d 2` first)
7. **`scripts/07_predict_validation_set.sh`** — inference (sliding-window + mirroring TTA, `--save_probabilities`) for all four trained configurations
8. **`scripts/08_fix_regions_class_order.py`** — patch `regions_class_order` in an existing `dataset.json` (if you have an *unfixed* Dataset002 already trained/predicted)
9. **`scripts/09_reconstruct_regions_from_probs.py`** — post-hoc label reconstruction from saved region probabilities using the corrected order, no retraining needed
10. **`scripts/10_postprocess_predictions.py <pred_dir> <out_dir>`** — volume-threshold filtering + hole-fill (no morphological opening) — run on every model's label maps
11. **`scripts/11_evaluate_cv.py <pred_dir> <gt_dir> [out.json]`** — internal DSC/NSD/lesion-wise-F1 evaluator (used for CV model selection; paper's reported numbers are the *official* scorer's output, see `results/`)
12. **`scripts/12_evaluate_all_folds.py`** — runs (11) across all 5 folds per model, prints 5-fold summary
13. **`scripts/13_convert_probs_to_region_format.py`** — converts a label-based model's 5-channel softmax into 4-channel WT/TC/ET/RC region-probability `.npy`, needed before ensembling
14. **`scripts/14_profile_lesion_sizes.py`** — per-case lesion count/size profiling, used to pick qualitative comparison cases (e.g. paper Figure 1)
15. **`scripts/15_ensemble_final.py`** — final probability-averaging ensemble (Primus label-based + ResEncL label-based only; region-based variants deliberately excluded, see script docstring), `MODE="cv"` to verify then `MODE="blind"` for the real submission

`configs/`:
- `dataset001_nnUNetPlans.json` — plain nnU-Net default plans (Dataset001)
- `dataset001_nnUNetResEncUNetLPlans.json` — ResEncL plans (Dataset001)
- `dataset002_nnUNetPlans.json` — default plans (Dataset002, region-based)
- `dataset002_dataset.json` — dataset.json with corrected `regions_class_order` already applied

Not included here (still only in the notebooks / your workspace, pull separately if you
want them in the repo too): `diagnose_tc_composition.py`, `check_rc_false_positive_rate.py`,
`check_reconstruction_sanity.py`, and the `visualize_comparison_paper.py` figure-generation
scripts used to build paper Figure 1.
