#!/usr/bin/env python
"""
script.py
Wrapper for train.py that allows passing parameters similar to deepspeed command line arguments.
Can optionally use DeepSpeed or regular training.
"""

import subprocess
import sys
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Wrapper for train.py with optional DeepSpeed support")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset_type", type=str, choices=['laion', 'imagenet'], default='laion',
                        help="Type of dataset to use: 'laion' for JSON-based or 'imagenet' for ImageNet-1k")
    parser.add_argument("--classified_json", type=str, default="classified_captions.jsonl",
                        help="Path to classified JSONL file (used when dataset_type is 'laion')")
    parser.add_argument("--imagenet_split", type=str, default="train",
                        help="ImageNet split to use when dataset_type is 'imagenet'")
    parser.add_argument("--max_imagenet_samples", type=int, default=None,
                        help="Maximum number of ImageNet samples to load (for debugging)")
    parser.add_argument("--upload_every", type=int, default=500)
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--deepspeed_config", type=str, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--org", type=str, required=True, help="Hugging Face organization or username")
    
    # Parse known args to handle additional parameters that might be passed to deepspeed
    args, unknown = parser.parse_known_args()
    
    # Build the command for train.py
    cmd = [
        sys.executable, "train.py",
        "--depth", str(args.depth),
        "--micro_batch_size", str(args.micro_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--lr", str(args.lr),
        "--dataset_type", args.dataset_type,
        "--classified_json", args.classified_json,
        "--upload_every", str(args.upload_every),
        "--org", args.org
    ]
    
    # Add ImageNet-specific arguments if needed
    if args.dataset_type == 'imagenet':
        cmd.extend(["--imagenet_split", args.imagenet_split])
        if args.max_imagenet_samples:
            cmd.extend(["--max_imagenet_samples", str(args.max_imagenet_samples)])
    
    # Add DeepSpeed related arguments if enabled
    if args.use_deepspeed:
        cmd.extend(["--use_deepspeed"])
        if args.deepspeed_config:
            cmd.extend(["--deepspeed_config", args.deepspeed_config])
    
    # Add local_rank if provided
    if args.local_rank != -1:
        cmd.extend(["--local_rank", str(args.local_rank)])
    
    # Add any additional unknown arguments that might be needed
    cmd.extend(unknown)
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the train.py script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
