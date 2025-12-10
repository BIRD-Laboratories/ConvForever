#!/usr/bin/env python
# script.py
# Wrapper script for train.py with optional DeepSpeed support

import subprocess
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Training script with optional DeepSpeed support')
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed for distributed training')
    parser.add_argument('--deepspeed_config', type=str, default='./ds_config.json', help='Path to DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    # Add other arguments that train.py accepts
    parser.add_argument('--depth', type=int, required=True, help='Depth of the ConvNeXt model')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='Micro batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--classified_json', type=str, default='classified_captions.jsonl', help='Path to classified JSONL file')
    parser.add_argument('--upload_every', type=int, default=500, help='Upload checkpoint every N steps')
    parser.add_argument('--org', type=str, required=True, help='Hugging Face organization or username')
    
    args = parser.parse_args()
    
    # Build the base command
    if args.deepspeed:
        # Use DeepSpeed
        cmd = [
            'deepspeed',
            '--num_gpus', '1',  # Adjust based on your setup
            'train.py'
        ]
        
        # Add DeepSpeed-specific arguments
        cmd.extend(['--deepspeed_config', args.deepspeed_config])
        cmd.extend(['--local_rank', str(args.local_rank)])
    else:
        # Run directly without DeepSpeed
        cmd = ['python', 'train.py']
    
    # Add other training arguments
    cmd.extend([
        '--depth', str(args.depth),
        '--micro_batch_size', str(args.micro_batch_size),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--lr', str(args.lr),
        '--classified_json', args.classified_json,
        '--upload_every', str(args.upload_every),
        '--org', args.org
    ])
    
    # Execute the command
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)
    except FileNotFoundError:
        if args.deepspeed:
            print("DeepSpeed not found. Make sure it's installed: pip install deepspeed")
        else:
            print("train.py not found. Make sure it exists in the current directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()