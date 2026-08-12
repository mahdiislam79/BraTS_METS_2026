#!/bin/bash
# 03_train_plain_nnunet.sh
# Preprocess + train the plain nnU-Net baseline on Dataset001 (label-based),
# default nnUNetTrainer (SGD, Nesterov momentum 0.99, initial LR 0.01,
# polynomial decay, per-sample Dice), default plans, 3d_fullres, 5 folds
# trained in parallel across 5 GPUs.

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# --- Preprocess ---
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres -np 8

# --- Train (5 folds in parallel, one per GPU) ---
nohup bash -c '
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$FOLD nnUNetv2_train 1 3d_fullres $FOLD \
        --npz \
        > /workspace/log_plainnnunet_fold${FOLD}.txt 2>&1 &
done
wait
echo "All 5 plain nnU-Net folds done" > /workspace/plainnnunet_status.txt
' > /workspace/nohup_plainnnunet_master.log 2>&1 &
echo "Launched in background, PID: $!"
