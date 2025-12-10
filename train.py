#!/usr/bin/env python
# train_convnext.py
# Load classified JSON, download images, train ConvNeXt with exact depth, upload checkpoints.

import argparse
import json
import logging
import os
import tempfile
import shutil
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import requests
from io import BytesIO
import deepspeed
from huggingface_hub import HfApi, ModelCard, ModelCardData
from tqdm import tqdm

# --- Category constants (must match classify_and_sync.py) ---
CATEGORIES = [
    "fish", "reptile", "amphibian", "bird", "mammal", "insect", "arachnid", "crustacean", "mollusk",
    "fungus", "flower", "fruit", "vegetable", "dish", "beverage", "bread", "meat", "dessert",
    "dog", "cat", "primate", "marsupial", "cetacean", "ungulate", "rodent", "raptor", "waterfowl",
    "songbird", "weapon", "tool", "vehicle", "watercraft", "aircraft", "instrument", "electronics",
    "furniture", "apparel", "footwear", "headwear", "accessory", "building", "structure", "fence",
    "bridge", "vessel", "container", "kitchenware", "sports", "geology", "landscape"
]
CATEGORY_SET = set(CATEGORIES)

def make_convnext_by_depth(depth: int, num_classes=1000, in_chans=3, **kwargs):
    if depth < 4:
        raise ValueError("Depth must be at least 4.")
    base = depth // 4
    extra = depth % 4
    depths = [base] * 4
    for i in range(extra):
        depths[i] += 1
    from timm.models.convnext import ConvNeXt
    dims = [96, 192, 384, 768]
    model = ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        drop_path_rate=kwargs.get("drop_path_rate", 0.1),
        **{k: v for k, v in kwargs.items() if k != "drop_path_rate"}
    )
    return model, depth

def download_image_to_path(url, temp_dir):
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
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        self.label_to_id = {cat: i for i, cat in enumerate(CATEGORIES)}
    def __len__(self): return len(self.records)
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
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def upload_to_hf(model, step, depth_num, org):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--classified_json", type=str, default="classified_captions.jsonl")
    parser.add_argument("--upload_every", type=int, default=500)
    parser.add_argument("--deepspeed", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--org", type=str, required=True, help="Hugging Face organization or username")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load classified data
    records = []
    with open(args.classified_json, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} classified records from {args.classified_json}")

    # Build model
    model, actual_depth = make_convnext_by_depth(args.depth, num_classes=len(CATEGORIES), drop_path_rate=0.1)
    logger.info(f"✅ Built ConvNeXt with exactly {actual_depth} layers")

    # DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=args.deepspeed,
        lr=args.lr
    )
    device = model_engine.device

    transform = get_transforms()
    global_step = 0

    # Train in batches
    eff_bs = args.micro_batch_size * args.gradient_accumulation_steps
    for i in range(0, len(records), eff_bs):
        batch_records = records[i:i + eff_bs]
        if len(batch_records) == 0:
            continue

        ds_batch = JsonImageDataset(batch_records, transform=transform)
        loader = DataLoader(ds_batch, batch_size=args.micro_batch_size, shuffle=True)

        model_engine.train()
        for imgs, labs in loader:
            if imgs is None or labs is None:
                continue
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model_engine(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labs)
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

        logger.info(f"Batch {i // eff_bs + 1}, Loss: {loss.item():.4f}")

        if global_step % args.upload_every == 0:
            base_model = getattr(model_engine, 'module', model_engine)
            upload_to_hf(base_model, global_step, actual_depth, args.org)

    # Final upload
    base_model = getattr(model_engine, 'module', model_engine)
    upload_to_hf(base_model, global_step, actual_depth, args.org)
    logger.info("✅ Training completed and model uploaded.")

if __name__ == "__main__":
    main()