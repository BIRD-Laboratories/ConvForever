#!/bin/bash

CSV_FILE="research/sweap.csv"

echo "Reading parameters from $CSV_FILE..."

# Read the CSV file, skipping the header line
tail -n +2 "$CSV_FILE" | while IFS=',' read -r approx_num_layers precision per_device_train_batch_size gradient_accumulation_steps enable_gradient_checkpointing optimizer learning_rate weight_decay lr_scheduler_type warmup_epochs drop_path_rate label_smoothing gradient_clipping; do
    echo "Row data: [$approx_num_layers], [$precision], [$per_device_train_batch_size], [$gradient_accumulation_steps], [$enable_gradient_checkpointing], [$optimizer], [$learning_rate], [$weight_decay], [$lr_scheduler_type], [$warmup_epochs], [$drop_path_rate], [$label_smoothing], [$gradient_clipping]"
    
    # Exit after first row for debugging
    break
done