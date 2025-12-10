"""
Utility functions for ConvForever
"""

import logging
import os
import tempfile
import torch
from huggingface_hub import HfApi, ModelCard, ModelCardData

from .model import CATEGORIES


def upload_to_hf(model, step, depth_num, org):
    """Upload model checkpoint to Hugging Face Hub."""
    model_name = f"ConvForever-{depth_num}-{step}"
    repo_id = f"{org}/{model_name}"
    
    with tempfile.TemporaryDirectory() as tmp:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(tmp, "pytorch_model.bin"))
            card = ModelCard.from_template(
                ModelCardData(license="apache-2.0", library_name="timm", tags=["convnext", "custom-depth"]),
                model_details=f"Custom ConvNeXt with exactly {depth_num} layers. Step {step}."
            )
            card.save(os.path.join(tmp, "README.md"))
            HfApi().create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
            HfApi().upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="model")
            logging.info(f"✅ Uploaded to https://huggingface.co/{repo_id}")


def download_image_to_path(url, temp_dir):
    """Download an image from URL to a temporary path."""
    import requests
    from io import BytesIO
    from PIL import Image
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        path = os.path.join(temp_dir, "img.jpg")
        img.save(path)
        return path
    except Exception:
        return None