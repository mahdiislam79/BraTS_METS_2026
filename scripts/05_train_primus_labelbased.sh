#!/bin/bash
# 05_train_primus_labelbased.sh
# Preprocess + train the Primus (PrimusV3S) transformer model, integrated as
# an nnU-Net v2 trainer class, on Dataset001 (label-based). Uses the
# trainer's own default optimization settings via nnUNet_PrimusV3S_Trainer.

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# --- Preprocess (shared with the plain nnU-Net baseline; skip if already done) ---
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres -np 8

# --- Train (5 folds in parallel, one per GPU) ---
nohup bash -c '
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$FOLD nnUNetv2_train 1 3d_fullres $FOLD \
        -tr nnUNet_PrimusV3S_Trainer --npz \
        > /workspace/log_primus_fold${FOLD}.txt 2>&1 &
done
wait
echo "All 5 Primus folds done" > /workspace/primus_status.txt
' > /workspace/nohup_primus_master.log 2>&1 &
echo "Launched in background, PID: $!"
