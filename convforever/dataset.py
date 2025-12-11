"""
Dataset utilities for ConvForever
"""

import os
import tempfile
import shutil
import json
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from .model import CATEGORIES


def download_image_to_path(url, temp_dir):
    """Download an image from URL to a temporary path."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        path = os.path.join(temp_dir, "img.jpg")
        img.save(path)
        return path
    except Exception:
        return None


class JsonImageDataset(Dataset):
    """Dataset class for loading JSON records with image URLs and labels."""
    
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        # Map all possible category labels to IDs
        self.label_to_id = {cat: i for i, cat in enumerate(CATEGORIES)}
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        rec = self.records[idx]
        temp_dir = tempfile.mkdtemp()
        img_path = download_image_to_path(rec["url"], temp_dir)
        if not img_path:
            shutil.rmtree(temp_dir)
            return self.__getitem__((idx + 1) % len(self))
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Handle both LAION-style and other label formats
        label = rec["label"]
        if label in self.label_to_id:
            label_id = self.label_to_id[label]
        else:
            # If label not in our predefined categories, assign a default or raise error
            # For now we'll skip this sample by returning next one
            return self.__getitem__((idx + 1) % len(self))
        
        return img, label_id


class ImageNetDataset(Dataset):
    """Dataset class for loading ImageNet-1k dataset from HuggingFace."""
    
    def __init__(self, split="train", transform=None, max_samples=None):
        """
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            transform: Transformations to apply to images
            max_samples: Maximum number of samples to load (for debugging)
        """
        # Load ImageNet dataset from HuggingFace
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        self.transform = transform
        
        # Get ImageNet class mapping (this maps numeric labels to descriptions)
        # Since we want to map to our custom categories, we'll need to create a mapping
        # For now, we'll use the raw ImageNet labels which are 1000 classes
        # Later we could potentially map to our categories if needed
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'].convert("RGB")  # Convert to RGB to ensure consistent format
        
        if self.transform:
            img = self.transform(img)
        
        # ImageNet uses numeric labels (0-999) for 1000 classes
        label_id = item['label']
        
        return img, label_id


def get_dataset(dataset_type, **kwargs):
    """
    Factory function to get the appropriate dataset based on type.
    
    Args:
        dataset_type: Type of dataset ('laion' or 'imagenet')
        **kwargs: Additional arguments for dataset initialization
    """
    if dataset_type.lower() == 'laion':
        return JsonImageDataset(kwargs['records'], transform=kwargs.get('transform'))
    elif dataset_type.lower() == 'imagenet':
        return ImageNetDataset(
            split=kwargs.get('split', 'train'),
            transform=kwargs.get('transform'),
            max_samples=kwargs.get('max_samples')
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_transforms():
    """Get standard image transformations for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])