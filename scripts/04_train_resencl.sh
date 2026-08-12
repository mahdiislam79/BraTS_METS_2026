#!/bin/bash
# 04_train_resencl.sh
# Plan, preprocess, and train the Residual Encoder Large (ResEncL) variant
# on Dataset001 (label-based). Same default nnUNetTrainer optimization
# protocol as the plain baseline, but with the ResEncL architecture plans.

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# --- Plan + preprocess with the ResEncL experiment planner ---
nnUNetv2_plan_experiment -d 1 -pl nnUNetPlannerResEncL
nnUNetv2_preprocess -d 1 -plans_name nnUNetResEncUNetLPlans -c 3d_fullres -np 8

# --- Train (5 folds in parallel, one per GPU) ---
nohup bash -c '
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$FOLD nnUNetv2_train 1 3d_fullres $FOLD \
        -p nnUNetResEncUNetLPlans --npz \
        > /workspace/log_resencl_fold${FOLD}.txt 2>&1 &
done
wait
echo "All 5 ResEncL folds done" > /workspace/resencl_status.txt
' > /workspace/nohup_resencl_master.log 2>&1 &
echo "Launched, PID: $!"
