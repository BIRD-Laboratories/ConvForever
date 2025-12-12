#!/usr/bin/env python
"""
train_convnext.py
Load classified JSON, download images, train ConvNeXt with exact depth, upload checkpoints.
"""

import argparse
import json
import logging
import torch
import torch.nn as nn

import convforever
from convforever import make_convnext_by_depth, JsonImageDataset, get_transforms, train_with_deepspeed, train_without_deepspeed, upload_to_hf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--classified_json", type=str, default="classified_captions.jsonl")
    parser.add_argument("--upload_every", type=int, default=500)
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--deepspeed_config", type=str, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--org", type=str, required=True, help="Hugging Face organization or username")
    parser.add_argument("--dataset_format", type=str, default="json", choices=["json", "laion", "imagenet", "pd_extended", "laion_imagenet"], help="Dataset format to use")
    parser.add_argument("--data_split", type=str, default="train", help="Data split to use (for non-JSON formats)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train (for non-JSON formats)")
    
    # Additional hyperparameters from sweep.csv
    parser.add_argument("--precision", type=str, default="fp32", help="Precision mode (fp32, bf16, etc.)")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay value")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop path rate for regularization")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping value")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load classified data or dataset based on format
    if args.dataset_format in ['json', 'laion']:
        # Load from JSON file
        records = []
        with open(args.classified_json, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} classified records from {args.classified_json}")
    else:
        # For other formats, we'll use the dataset directly
        records = None
        logger.info(f"Using dataset format: {args.dataset_format}, split: {args.data_split}")

    # Build model
    model, actual_depth = make_convnext_by_depth(args.depth, num_classes=len(convforever.model.CATEGORIES), drop_path_rate=args.drop_path_rate)
    logger.info(f"✅ Built ConvNeXt with exactly {actual_depth} layers")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transforms()

    # Choose training method based on DeepSpeed flag
    if args.use_deepspeed:
        if args.deepspeed_config is None:
            raise ValueError("--deepspeed_config is required when using --use_deepspeed")
        model_trained, final_step = train_with_deepspeed(model, args, device, records, transform)
    else:
        model_trained, final_step = train_without_deepspeed(model, args, device, records, transform)

    # Final upload
    base_model = getattr(model_trained, 'module', model_trained)
    upload_to_hf(base_model, final_step, actual_depth, args.org)
    logger.info("✅ Training completed and model uploaded.")


if __name__ == "__main__":
    main()