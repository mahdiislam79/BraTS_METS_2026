# Brain Metastases Segmentation for BraTS 2026 Task 1: A Multi-Architecture Comparison

Code accompanying our BraTS-METS 2026 Task 1 short paper comparing a plain nnU-Net baseline, a Residual Encoder Large (ResEncL) variant, region-based training, and a Primus (PrimusV3S) transformer model for brain metastasis segmentation.

**Paper:** [link to OpenReview / camera-ready PDF once available]
**Challenge:** [BraTS-METS 2026 Task 1](https://www.synapse.org/#!Synapse:syn75814152) (Synapse ID: `syn75814152`)

---

## What's in this repository

This repo contains **configuration files, training/evaluation/postprocessing scripts, and Docker submission code** — not model weights or patient data. See [Data Availability](#data-availability) and [Model Weights](#model-weights) below for how to obtain those separately.

```
.
├── configs/                # nnU-Net plans and dataset.json for all three dataset variants
├── scripts/                # training launch scripts, CV evaluation, postprocessing, ensembling
├── results/                # official BraTS-METS validation-server scores (per-case CSVs)
├── docker/                 # containerized inference submission (Dockerfile + entrypoint)
└── paper/                  # LaTeX source for the short paper
```

### `configs/`
- `dataset.json` — 4-modality (T1/T1CE/T2/FLAIR), 5-class label scheme (background, NETC, SNFH, ET, RC), and the corrected `regions_class_order: [2, 1, 3, 4]` (outermost-first) used for region-based label reconstruction.
- `nnUNetPlans.json` — default nnU-Net v2 plans, Dataset001 (label-based, plain nnU-Net baseline).
- `nnUNetResEncUNetLPlans.json` — Residual Encoder Large plans, Dataset001.
- `regions_nnUNetPlans.json` — default plans for Dataset002 (region-based / overlapping-label variant).

### `scripts/`
- `train_resencl.sh`, `train_primus_regions.sh` — the exact 5-fold parallel training launch commands used (nnU-Net v2 CLI, one fold per GPU).
- `evaluate_cv.py` — computes lesion-wise DSC, NSD (τ=2.0mm), and lesion-wise F1, all applying a consistent 27mm³ connected-component volume floor (6-connectivity, `scipy.ndimage.label` default) across all three metrics. Uses greedy any-voxel-overlap lesion matching (not optimal bipartite matching). Used internally for cross-validation model selection and ablation checks prior to submission; the paper's reported numbers come from the official server (`results/`), not this script.
- `postprocess_predictions.py` — volume-threshold (≥27mm³) connected-component filtering + per-class hole-filling. **Deliberately excludes `scipy.ndimage.binary_opening`** — see paper Section 2.6 / this repo's Known Pitfalls below for why.
- `ensemble_final.py` — weighted probability-averaging ensemble across models, converting label-based softmax outputs to region probabilities before combination, followed by the same postprocessing pipeline.
- `fix_regions_class_order.py` — post-hoc label reconstruction fix for predictions generated under the incorrect ascending `regions_class_order`, without requiring retraining.

### `results/`
Per-case official validation scores as returned by the BraTS-METS validation server, one CSV per submission (`all_instance`, `large_instance`, and `small_instance` TP/FP/FN/precision/recall/F1, plus `lesionwise_dsc_mean`/`lesionwise_nsd_mean`/`lesionwise_hd95_mean`, per class, per case), each with `mean`/`std`/`median` summary rows. **All DSC, NSD, and lesion-wise F1 values reported in the paper's Tables 1 and 2 are taken directly from these files** — not recomputed by `evaluate_cv.py`, which was used only for internal cross-validation model selection prior to submission. Filenames indicate the submitted configuration (e.g. `all_scores_resencl_label.csv`, `all_scores_ensemble.csv`).

### `docker/`
Containerized inference submission built for the challenge's evaluation environment (NVIDIA A10G 24GB, offline, 12-hour budget). Built with `docker buildx build --platform linux/amd64` (required when building on Apple Silicon, since the evaluator runs on x86_64).

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
3. **Train** using the scripts in `scripts/` (adjust `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` paths for your environment).
4. **Predict** with `nnUNetv2_predict` using the corresponding trainer/plans/fold.
5. **Postprocess** with `scripts/postprocess_predictions.py` (volume threshold + hole-fill only).
6. **Ensemble** (optional) with `scripts/ensemble_final.py`.
7. **Evaluate** with `scripts/evaluate_cv.py <pred_dir> <gt_dir>`.

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

[Choose and state a license here — MIT or Apache-2.0 are common for challenge submission code. Note that the dataset itself remains under its own separate Data Use Agreement regardless of this repo's code license.]

## Disclosure of Interests

The authors have no competing interests to declare that are relevant to the content of this work.
