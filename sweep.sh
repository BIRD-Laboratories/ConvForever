#!/bin/bash

# Script to read parameters from sweap.csv and run training with them
CSV_FILE="research/sweap.csv"

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: $CSV_FILE does not exist!"
    exit 1
fi

echo "Reading parameters from $CSV_FILE and running training..."

# Read the CSV file, skipping the header line
tail -n +2 "$CSV_FILE" | while IFS=',' read -r approx_num_layers precision per_device_train_batch_size gradient_accumulation_steps enable_gradient_checkpointing optimizer learning_rate weight_decay lr_scheduler_type warmup_epochs drop_path_rate label_smoothing gradient_clipping; do
    
    # Skip empty lines
    if [[ -z "$approx_num_layers" ]]; then
        continue
    fi
    
    echo "========================================="
    echo "Running training with parameters:"
    echo "  approx_num_layers: $approx_num_layers"
    echo "  precision: $precision"
    echo "  per_device_train_batch_size: $per_device_train_batch_size"
    echo "  gradient_accumulation_steps: $gradient_accumulation_steps"
    echo "  enable_gradient_checkpointing: $enable_gradient_checkpointing"
    echo "  optimizer: $optimizer"
    echo "  learning_rate: $learning_rate"
    echo "  weight_decay: $weight_decay"
    echo "  lr_scheduler_type: $lr_scheduler_type"
    echo "  warmup_epochs: $warmup_epochs"
    echo "  drop_path_rate: $drop_path_rate"
    echo "  label_smoothing: $label_smoothing"
    echo "  gradient_clipping: $gradient_clipping"
    echo "========================================="
    
    # Convert CSV values to appropriate argument names for train.py
    # Map approx_num_layers to depth
    # Map per_device_train_batch_size to micro_batch_size
    # Convert enable_gradient_checkpointing to boolean flag if true
    GRAD_CHECKPOINT_ARG=""
    if [[ "$enable_gradient_checkpointing" == "True" || "$enable_gradient_checkpointing" == "true" ]]; then
        GRAD_CHECKPOINT_ARG="--enable_gradient_checkpointing"
    fi
    
    echo "Executing training command..."
    python train.py \
        --depth "$approx_num_layers" \
        --micro_batch_size "$per_device_train_batch_size" \
        --gradient_accumulation_steps "$gradient_accumulation_steps" \
        --lr "$learning_rate" \
        --precision "$precision" \
        $GRAD_CHECKPOINT_ARG \
        --optimizer "$optimizer" \
        --weight_decay "$weight_decay" \
        --lr_scheduler_type "$lr_scheduler_type" \
        --warmup_epochs "$warmup_epochs" \
        --drop_path_rate "$drop_path_rate" \
        --label_smoothing "$label_smoothing" \
        --gradient_clipping "$gradient_clipping" \
        --org "your-org-here" \
        --dataset_format "json" \
        --epochs 1 \
        --upload_every 500
    
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for this parameter set."
    else
        echo "Training failed for this parameter set."
    fi
    
    echo "Test completed for this parameter set."
    echo ""
done

echo "All parameter sets have been processed."