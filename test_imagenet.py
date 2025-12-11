#!/usr/bin/env python3
"""
Test script for ImageNet dataset integration
"""

from convforever.dataset import get_imagenet_dataloader
import torch

def test_imagenet_dataloader():
    print("Testing ImageNet dataloader...")
    
    try:
        # Create a small dataloader for testing (just a few samples)
        # Using smaller batch size and fewer workers for testing
        dataloader = get_imagenet_dataloader(
            split='train', 
            batch_size=2, 
            shuffle=False, 
            num_workers=1
        )
        
        print(f"Dataloader created successfully")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Number of workers: {dataloader.num_workers}")
        
        # Get one batch to test
        for i, (images, labels) in enumerate(dataloader):
            print(f"Batch {i}: Images shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"Labels: {labels}")
            
            if i >= 0:  # Just test the first batch
                break
        
        print("ImageNet dataloader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during ImageNet dataloader test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imagenet_dataloader()