#!/bin/bash

# Script to read parameters from sweap.csv and test them
CSV_FILE="research/sweap.csv"

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: $CSV_FILE does not exist!"
    exit 1
fi

echo "Reading parameters from $CSV_FILE and testing them..."

# Read the CSV file, skipping the header line
tail -n +2 "$CSV_FILE" | while IFS=',' read -r approx_num_layers precision per_device_train_batch_size gradient_accumulation_steps enable_gradient_checkpointing optimizer learning_rate weight_decay lr_scheduler_type warmup_epochs drop_path_rate label_smoothing gradient_clipping; do
    
    # Skip empty lines
    if [[ -z "$approx_num_layers" ]]; then
        continue
    fi
    
    echo "========================================="
    echo "Testing parameters:"
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
    
    # Here you would add your actual testing logic
    # For now, I'll include example commands that demonstrate how to use these parameters
    echo "Running test with these parameters..."
    
    # Example of how you might use these parameters in a real training command:
    # python train_model.py \
    #     --approx_num_layers="$approx_num_layers" \
    #     --precision="$precision" \
    #     --batch_size="$per_device_train_batch_size" \
    #     --grad_acc_steps="$gradient_accumulation_steps" \
    #     --checkpointing="$enable_gradient_checkpointing" \
    #     --optimizer="$optimizer" \
    #     --lr="$learning_rate" \
    #     --weight_decay="$weight_decay" \
    #     --scheduler="$lr_scheduler_type" \
    #     --warmup_epochs="$warmup_epochs" \
    #     --drop_path_rate="$drop_path_rate" \
    #     --label_smoothing="$label_smoothing" \
    #     --grad_clip="$gradient_clipping"
    
    # Example: Print out a sample command that could be executed
    echo "Sample command that would be executed:"
    echo "train_model --layers=$approx_num_layers --precision=$precision --batch-size=$per_device_train_batch_size --accum-steps=$gradient_accumulation_steps --lr=$learning_rate"
    
    # Add a delay to simulate processing time (remove this in actual implementation)
    sleep 1
    
    echo "Test completed for this parameter set."
    echo ""
done

echo "All parameter sets have been processed."