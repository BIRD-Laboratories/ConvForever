"""
Dataset utilities for ConvForever
"""

import tempfile
import shutil
import json
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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
        
        label_id = self.label_to_id[rec["label"]]
        return img, label_id


def get_transforms():
    """Get standard image transformations for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])