#!/usr/bin/env python3
"""
Example usage of both ImageNet and LAION format support
"""

from convforever.dataset import get_dataset_by_format

def example_imagenet_usage():
    """Example of using the standard ImageNet format"""
    print("=== Using Standard ImageNet Format ===")
    
    # Create training dataloader
    train_loader = get_dataset_by_format(
        dataset_format='imagenet',
        split='train',
        batch_size=32,
        shuffle=True
    )
    
    # Create validation dataloader  
    val_loader = get_dataset_by_format(
        dataset_format='imagenet',
        split='validation',
        batch_size=32,
        shuffle=False
    )
    
    print(f"Training dataset: {len(train_loader.dataset)} samples")
    print(f"Validation dataset: {len(val_loader.dataset)} samples")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")


def example_laion_usage():
    """Example of using the LAION JSON format"""
    print("\n=== Using LAION JSON Format ===")
    
    # This would be used with a JSON file containing URL-label pairs
    # json_loader = get_dataset_by_format(
    #     dataset_format='laion',
    #     split='/path/to/your/dataset.json',  # Path to your JSON file
    #     batch_size=32,
    #     shuffle=True
    # )
    #
    # print(f"JSON dataset: {len(json_loader.dataset)} samples")
    # print(f"Batch shape: {next(iter(json_loader))[0].shape}")
    
    print("Example setup for LAION format:")
    print("""
json_loader = get_dataset_by_format(
    dataset_format='laion',  # or 'json'
    split='/path/to/dataset.json',  # Path to your JSON file
    batch_size=32,
    shuffle=True
)
""")


def example_laion_imagenet_usage():
    """Example of using the LAION-style ImageNet format"""
    print("\n=== Using LAION-Style ImageNet Format ===")
    
    # Create dataloader with possible custom label mapping
    loader = get_dataset_by_format(
        dataset_format='laion_imagenet',
        split='validation',
        batch_size=32,
        shuffle=False
    )
    
    print(f"LAION-ImageNet dataset: {len(loader.dataset)} samples")
    print(f"Batch shape: {next(iter(loader))[0].shape}")


def main():
    print("Dataset Format Usage Examples")
    print("=" * 50)
    
    example_imagenet_usage()
    example_laion_usage()
    example_laion_imagenet_usage()
    
    print("\nFor JSON format, create a file with structure:")
    print("""
[
    {
        "url": "https://example.com/image1.jpg",
        "label": "dog"
    },
    {
        "url": "https://example.com/image2.jpg", 
        "label": "cat"
    }
]
""")


if __name__ == "__main__":
    main()