"""
ConvForever - Scaling Convolutional Neural Networks Forever
"""

from .model import make_convnext_by_depth
from .dataset import JsonImageDataset, get_transforms
from .trainer import train_with_deepspeed, train_without_deepspeed
from .utils import upload_to_hf, download_image_to_path

__version__ = "1.0.0"
__author__ = "Julian Herrera"

__all__ = [
    "make_convnext_by_depth",
    "JsonImageDataset", 
    "get_transforms",
    "train_with_deepspeed",
    "train_without_deepspeed",
    "upload_to_hf",
    "download_image_to_path"
]