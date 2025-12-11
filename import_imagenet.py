#!/usr/bin/env python
"""
Import ImageNet-1k dataset from HuggingFace.
This script loads the ImageNet dataset and validates it for training.
"""

import argparse
import logging
from convforever.dataset import ImageNetDataset


def main():
    parser = argparse.ArgumentParser(description="Import ImageNet-1k dataset from HuggingFace")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to load ('train', 'validation', or 'test')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to load (for testing)")
    parser.add_argument("--output_info", type=str, default="imagenet_info.json",
                        help="Path to save dataset information")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading ImageNet dataset split: {args.split}")
    
    try:
        # Create ImageNet dataset
        imagenet_dataset = ImageNetDataset(
            split=args.split,
            transform=None,  # No transforms for import validation
            max_samples=args.max_samples
        )
        
        logger.info(f"Successfully loaded ImageNet dataset with {len(imagenet_dataset)} samples")
        
        # Print some basic info about the dataset
        logger.info(f"Dataset size: {len(imagenet_dataset)}")
        
        if len(imagenet_dataset) > 0:
            # Get a sample to verify
            sample_img, sample_label = imagenet_dataset[0]
            logger.info(f"Sample image shape: {sample_img.size}")
            logger.info(f"Sample label: {sample_label}")
        
        # Save dataset info
        import json
        info = {
            "split": args.split,
            "size": len(imagenet_dataset),
            "max_samples": args.max_samples
        }
        
        with open(args.output_info, "w") as f:
            json.dump(info, f)
        
        logger.info(f"Dataset info saved to {args.output_info}")
        
    except Exception as e:
        logger.error(f"Error loading ImageNet dataset: {e}")
        raise


if __name__ == "__main__":
    main()