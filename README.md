# Brain Metastases Segmentation for BraTS 2026 Task 1: A Multi-Architecture Comparison

Code accompanying our BraTS-METS 2026 Task 1 short paper comparing a plain nnU-Net baseline, a Residual Encoder Large (ResEncL) variant, region-based training, and a Primus (PrimusV3S) transformer model for brain metastasis segmentation.

**Repository:** https://github.com/mahdiislam79/BraTS_METS_2026
**Paper:** [link to OpenReview / camera-ready PDF once available]
**Challenge:** [BraTS-METS 2026 Task 1](https://www.synapse.org/#!Synapse:syn75814152) (Synapse ID: `syn75814152`)

---

## What's in this repository

This repo contains **configuration files, training/evaluation/postprocessing scripts, and Docker submission code** — not model weights or patient data. See [Data Availability](#data-availability) and [Model Weights](#model-weights) below for how to obtain those separately.

```
.
├── configs/                # nnU-Net plans and dataset.json for all three dataset variants
├── scripts/                # training launch scripts, CV evaluation, postprocessing, ensembling (numbered in pipeline order)
├── results/                # official BraTS-METS validation-server scores (per-case CSVs)
├── paper/                  # LaTeX source for the short paper
└── RUN_ORDER.md            # step-by-step description of what each script in scripts/ does and when to run it
```

### `configs/`
- `dataset002_dataset.json` — 4-modality (T1/T1CE/T2/FLAIR) region-based dataset.json, with the corrected `regions_class_order: [2, 1, 3, 4]` (outermost-first) already applied.
- `dataset001_nnUNetPlans.json` — default nnU-Net v2 plans, Dataset001 (label-based, plain nnU-Net baseline).
- `dataset001_nnUNetResEncUNetLPlans.json` — Residual Encoder Large plans, Dataset001.
- `dataset002_nnUNetPlans.json` — default plans for Dataset002 (region-based / overlapping-label variant).

### `scripts/`
Numbered in pipeline order — see `RUN_ORDER.md` for the full step-by-step description of each file. Summary of the key ones:
- `01_convert_dataset_to_nnunet.py` — raw BraTS-METS data → Dataset001 (label-based) nnU-Net raw format.
- `02_create_patient_grouped_splits.py` — builds the patient-grouped 5-fold `splits_final.json` used identically across every architecture.
- `03_train_plain_nnunet.sh`, `04_train_resencl.sh`, `05_train_primus_labelbased.sh`, `06_build_regions_dataset_and_train_primus.py` + `06b_train_primus_regionbased.sh` — the exact preprocessing + 5-fold parallel training launch commands used for each configuration (nnU-Net v2 CLI, one fold per GPU).
- `07_predict_validation_set.sh` — sliding-window + mirroring TTA inference with `--save_probabilities` for all four trained configurations.
- `08_fix_regions_class_order.py` — patches `regions_class_order` in an existing `dataset.json`.
- `09_reconstruct_regions_from_probs.py` — post-hoc label reconstruction fix for predictions generated under the incorrect ascending `regions_class_order`, without requiring retraining.
- `10_postprocess_predictions.py` — volume-threshold (≥27mm³) connected-component filtering + per-class hole-filling. **Deliberately excludes `scipy.ndimage.binary_opening`** — see paper Section 2.6 / this repo's Known Pitfalls below for why.
- `11_evaluate_cv.py` — computes lesion-wise DSC, NSD (τ=2.0mm), and lesion-wise F1, all applying a consistent 27mm³ connected-component volume floor (6-connectivity, `scipy.ndimage.label` default) across all three metrics. Uses greedy any-voxel-overlap lesion matching (not optimal bipartite matching). Used internally for cross-validation model selection and ablation checks prior to submission; the paper's reported numbers come from the official server (`results/`), not this script.
- `12_evaluate_all_folds.py` — runs (11) across all 5 CV folds per model configuration.
- `13_convert_probs_to_region_format.py` — converts a label-based model's 5-channel softmax into 4-channel WT/TC/ET/RC region-probability format, needed before ensembling.
- `14_profile_lesion_sizes.py` — per-case lesion count/size profiling, used to select qualitative comparison cases (e.g. paper Figure 1).
- `15_ensemble_final.py` — weighted probability-averaging ensemble (Primus label-based + ResEncL label-based only — both region-based variants are deliberately excluded; see the script's docstring for why), with `MODE="cv"` for verification and `MODE="blind"` for the real submission.

### `results/`
Per-case official validation scores as returned by the BraTS-METS validation server, one CSV per submission (`all_instance`, `large_instance`, and `small_instance` TP/FP/FN/precision/recall/F1, plus `lesionwise_dsc_mean`/`lesionwise_nsd_mean`/`lesionwise_hd95_mean`, per class, per case), each with `mean`/`std`/`median` summary rows. **All DSC, NSD, and lesion-wise F1 values reported in the paper's Tables 1 and 2 are taken directly from these files** — not recomputed by `evaluate_cv.py`, which was used only for internal cross-validation model selection prior to submission. Filenames indicate the submitted configuration (e.g. `all_scores_resencl_label.csv`, `all_scores_ensemble.csv`).

### `paper/`
LaTeX source (Springer LNCS format) for the short paper submitted to BraTS-METS 2026 Task 1.

---

## Data Availability

This project uses the **BraTS-METS 2025 Lighthouse dataset** (reused for BraTS 2026 Task 1), comprising 1,496 cases from nine institutions. The dataset is **not redistributed in this repository** — it is available only under the official challenge Data Use Agreement via Synapse:

- Synapse ID: `syn74274097`
- Registration required: https://www.synapse.org/#!Synapse:syn74274097

## Model Weights

Trained model checkpoints are not included in this repository due to size. [Add a link here if/when weights are published to Zenodo, Hugging Face, or a Synapse Docker repository, e.g. `docker.synapse.org/syn75814152/bratsmets-ensemble:v1`.]

---

## Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `nnunetv2`, `torch`, `scipy`, `nibabel`, `surface-distance` (for NSD computation — install via `pip install surface-distance`; NSD scores will silently be skipped if this package is absent).

Training was run on NVIDIA H100 SXM (RunPod) and NVIDIA A100 (VSC cluster) GPUs. Inference/Docker evaluation targets NVIDIA A10G (24GB VRAM).

---

## Reproducing the pipeline

1. **Convert data** to nnU-Net raw format (`imagesTr`/`labelsTr`) following the official BraTS-METS conversion guidance, using `configs/dataset.json` as the label scheme reference.
2. **Preprocess:**
   ```bash
   nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity   # label-based
   nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity   # region-based
   ```
3. **Train** using `scripts/03_train_plain_nnunet.sh` through `06b_train_primus_regionbased.sh` (adjust `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` paths for your environment — these are hardcoded to the original `/workspace/...` training environment).
4. **Predict** with `scripts/07_predict_validation_set.sh`.
5. **Postprocess** with `scripts/10_postprocess_predictions.py <pred_dir> <out_dir>` (volume threshold + hole-fill only).
6. **Ensemble** (optional) with `scripts/15_ensemble_final.py`.
7. **Evaluate** with `scripts/11_evaluate_cv.py <pred_dir> <gt_dir>`.

See `RUN_ORDER.md` for the complete numbered pipeline, including the region-based dataset construction and reconstruction-order fix steps.

---

## Known Pitfalls (documented so others don't repeat them)

1. **Morphological opening destroys small lesions.** Applying `binary_opening` before volume-threshold filtering erodes small (~27mm³-scale) components before dilation can restore them, collapsing lesion-wise F1 from ~0.4 to ~0.01 despite near-unchanged Dice. Use volume-threshold filtering + hole-filling only.
2. **Region reconstruction order must match anatomical nesting, not ascending label value.** nnU-Net's region-based reconstruction overwrites in declared order; our label scheme (NETC=1 innermost, SNFH=2 outermost) is the reverse of ascending order. The correct `regions_class_order` here is `[2, 1, 3, 4]`. This must be verified independently in every environment/config copy — nnU-Net keeps a duplicate `dataset.json` inside each trainer's results folder, and fixing only the `nnUNet_raw` copy does not propagate to inference.
3. **Probability-averaging ensembles do not automatically inherit a component model's minority-class strengths**, even when weighted to protect them, due to the fixed-threshold suppression effect described in the paper's Discussion section.

---

## Citation

```bibtex
@inproceedings{islam2026bratsmets,
  title     = {Brain Metastases Segmentation for BraTS 2026 Task 1: A Multi-Architecture Comparison},
  author    = {Islam, Mahdi and Tabassum, Musarrat},
  booktitle = {BraTS-METS 2026 Challenge, MICCAI},
  year      = {2026}
}
```

## Acknowledgments

Data used in this publication were obtained as part of the BraTS-METS Challenge project through Synapse ID `syn74274097`.

## License

This repository's code is released under the MIT License. Note that the dataset itself remains under its own separate Data Use Agreement regardless of this repo's code license (see Data Availability above).

## Disclosure of Interests

The authors have no competing interests to declare that are relevant to the content of this work.
