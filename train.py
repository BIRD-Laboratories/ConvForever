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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load classified data
    records = []
    with open(args.classified_json, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} classified records from {args.classified_json}")

    # Build model
    model, actual_depth = make_convnext_by_depth(args.depth, num_classes=len(convforever.model.CATEGORIES), drop_path_rate=0.1)
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