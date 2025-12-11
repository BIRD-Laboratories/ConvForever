#!/usr/bin/env python3
"""
Test script to verify both ImageNet and LAION format support
"""

import os
import json
from convforever.dataset import get_dataset_by_format

def create_sample_json():
    """Create a sample JSON file in LAION format for testing"""
    sample_data = [
        {
            "url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1",  # Example bird image
            "label": "bird"
        },
        {
            "url": "https://images.unsplash.com/photo-1535982330223-302676b6b6ab",  # Example dog image
            "label": "dog"
        }
    ]
    
    with open('/workspace/sample_data.json', 'w') as f:
        json.dump(sample_data, f)
    
    return '/workspace/sample_data.json'

def test_imagenet_format():
    """Test the standard ImageNet format"""
    print("Testing ImageNet format...")
    try:
        # Test with a small subset - just get the validation split info
        dataloader = get_dataset_by_format(
            dataset_format='imagenet',
            split='validation',
            batch_size=4,
            shuffle=False
        )
        print(f"✓ ImageNet format: Successfully created dataloader with {len(dataloader.dataset)} samples")
        print(f"  Sample batch shape: {next(iter(dataloader))[0].shape}")
    except Exception as e:
        print(f"✗ ImageNet format failed: {e}")

def test_laion_format():
    """Test the LAION JSON format"""
    print("\nTesting LAION JSON format...")
    try:
        # Create sample JSON file
        json_path = create_sample_json()
        
        # Test with the sample JSON file
        dataloader = get_dataset_by_format(
            dataset_format='laion',
            split=json_path,
            batch_size=2,
            shuffle=False
        )
        print(f"✓ LAION JSON format: Successfully created dataloader with {len(dataloader.dataset)} samples")
        print(f"  Sample batch shape: {next(iter(dataloader))[0].shape}")
    except Exception as e:
        print(f"✗ LAION JSON format failed: {e}")

def test_laion_imagenet_format():
    """Test the LAION-style ImageNet format"""
    print("\nTesting LAION-style ImageNet format...")
    try:
        dataloader = get_dataset_by_format(
            dataset_format='laion_imagenet',
            split='validation',
            batch_size=4,
            shuffle=False
        )
        print(f"✓ LAION-style ImageNet format: Successfully created dataloader with {len(dataloader.dataset)} samples")
        print(f"  Sample batch shape: {next(iter(dataloader))[0].shape}")
    except Exception as e:
        print(f"✗ LAION-style ImageNet format failed: {e}")

if __name__ == "__main__":
    print("Testing dataset format support...\n")
    
    test_imagenet_format()
    test_laion_format()
    test_laion_imagenet_format()
    
    print("\nAll tests completed!")