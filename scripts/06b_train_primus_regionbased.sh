#!/bin/bash
# 06b_train_primus_regionbased.sh
# Trains Primus (nnUNet_PrimusV3S_Trainer) on Dataset002 (region-based).
# Run 06_build_regions_dataset_and_train_primus.py and
# nnUNetv2_plan_and_preprocess -d 2 first.

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

nohup bash -c '
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$FOLD nnUNetv2_train 2 3d_fullres $FOLD \
        -tr nnUNet_PrimusV3S_Trainer --npz \
        > /workspace/log_primus_regions_fold${FOLD}.txt 2>&1 &
done
wait
echo "All 5 Primus-regions folds done" > /workspace/primus_regions_status.txt
' > /workspace/nohup_primus_regions_master.log 2>&1 &
echo "Launched, PID: $!"
