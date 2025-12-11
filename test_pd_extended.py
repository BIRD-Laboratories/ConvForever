#!/usr/bin/env python3
"""
Test script for PD Extended dataset integration
"""

from convforever.dataset import get_dataset_by_format
import torch

def test_pd_extended_dataloader():
    print("Testing PD Extended dataloader...")
    
    try:
        # Create a small dataloader for testing (just a few samples)
        # Using smaller batch size and fewer workers for testing
        dataloader = get_dataset_by_format(
            dataset_format='pd_extended', 
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
        
        print("PD Extended dataloader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during PD Extended dataloader test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pd_extended_dataloader()