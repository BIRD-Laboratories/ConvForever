"""
ConvForever - Scaling Convolutional Neural Networks Forever
"""

from .model import make_convnext_by_depth, CATEGORIES
from .dataset import JsonImageDataset, ImageNetDataset, get_transforms, get_dataset
from .trainer import train_with_deepspeed, train_without_deepspeed
from .utils import upload_to_hf

__version__ = "1.0.0"
__author__ = "Julian Herrera"

__all__ = [
    "make_convnext_by_depth",
    "CATEGORIES",
    "JsonImageDataset", 
    "ImageNetDataset",
    "get_dataset",
    "get_transforms",
    "train_with_deepspeed",
    "train_without_deepspeed",
    "upload_to_hf"
]