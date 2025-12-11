# Dataset Format Support

This project supports multiple dataset formats for flexibility in training models. Below are the supported formats and how to use them.

## Supported Formats

### 1. Standard ImageNet Format

Uses the official ImageNet-1K dataset from Hugging Face.

```python
from convforever.dataset import get_dataset_by_format

# Get ImageNet training data
train_loader = get_dataset_by_format(
    dataset_format='imagenet',
    split='train',
    batch_size=32
)

# Get ImageNet validation data
val_loader = get_dataset_by_format(
    dataset_format='imagenet',
    split='validation',
    batch_size=32
)
```

### 2. LAION Format (JSON)

Supports datasets in JSON format with image URLs and labels, similar to LAION datasets.

```python
from convforever.dataset import get_dataset_by_format

# Get data from a JSON file with URL and label pairs
json_loader = get_dataset_by_format(
    dataset_format='laion',  # or 'json'
    split='/path/to/your/dataset.json',  # Path to JSON file
    batch_size=32
)
```

The JSON file should contain entries in the following format:
```json
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
```

### 3. LAION-Style ImageNet Format

A variant that uses the Hugging Face ImageNet dataset but allows for custom label mappings.

```python
from convforever.dataset import get_dataset_by_format

# Get ImageNet data with custom label mapping
custom_loader = get_dataset_by_format(
    dataset_format='laion_imagenet',
    split='train',
    batch_size=32,
    label_mapping_file='/path/to/label_mapping.json'  # Optional
)
```

## Unified Access Function

The `get_dataset_by_format()` function provides unified access to all supported formats:

```python
def get_dataset_by_format(
    dataset_format='imagenet',     # 'imagenet', 'laion', 'json', or 'laion_imagenet'
    split='train',                 # For ImageNet: 'train', 'validation', 'test'. For JSON: path to file
    batch_size=32,
    shuffle=True,
    transform=None,                # Custom transforms, otherwise uses defaults
    num_workers=4,
    **kwargs                     # Additional arguments for specific formats
):
    pass
```

## Features

- **Automatic Transform Selection**: Training vs validation transforms applied automatically
- **Flexible Label Handling**: Supports both string labels (LAION format) and numeric labels (ImageNet format)
- **Error Handling**: Graceful handling of failed image downloads in JSON format
- **Memory Efficient**: Uses temporary directories for downloaded images in JSON format
- **Extensible**: Easy to add new dataset formats