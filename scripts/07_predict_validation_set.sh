#!/bin/bash
# 07_predict_validation_set.sh
# Runs nnU-Net's default sliding-window inference with mirroring TTA,
# saving softmax probabilities (--save_probabilities), for each trained
# model on the blind validation set (imagesTs).

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# --- Plain nnU-Net baseline (Dataset001, default plans/trainer) ---
nnUNetv2_predict \
    -i "${nnUNet_raw}/Dataset001_BraTSMETS2026/imagesTs" \
    -o "/workspace/predictions/predictions_plainnnunet_validation" \
    -d 1 -c 3d_fullres -f 0 1 2 3 4 \
    --save_probabilities -npp 2 -nps 2

# --- ResEncL (Dataset001, nnUNetResEncUNetLPlans) ---
nnUNetv2_predict \
    -i "${nnUNet_raw}/Dataset001_BraTSMETS2026/imagesTs" \
    -o "/workspace/predictions/predictions_resencl_labelbased_validation" \
    -d 1 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans -c 3d_fullres -f 0 1 2 3 4 \
    --save_probabilities

# --- Primus label-based (Dataset001) ---
nnUNetv2_predict \
    -i "${nnUNet_raw}/Dataset001_BraTSMETS2026/imagesTs" \
    -o "/workspace/predictions_primus_validation" \
    -d 1 -c 3d_fullres -tr nnUNet_PrimusV3S_Trainer -f 0 1 2 3 4 \
    --save_probabilities -npp 2 -nps 2

# --- Primus region-based (Dataset002) ---
nnUNetv2_predict \
    -i "${nnUNet_raw}/Dataset002_BraTSMETS2026Regions/imagesTs" \
    -o "/workspace/predictions_primus_regions_validation" \
    -d 2 -c 3d_fullres -tr nnUNet_PrimusV3S_Trainer -f 0 1 2 3 4 \
    --save_probabilities -npp 2 -nps 2

echo "All predictions written. Next: run 08_fix_regions_class_order.py + "
echo "09_reconstruct_regions_from_probs.py on the region-based output, then "
echo "10_postprocess_predictions.py on every model's label maps."
