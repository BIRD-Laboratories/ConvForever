#!/usr/bin/env python
# Train a ConvNeXt model with EXACTLY --depth N layers (no presets)
# Real-time relaion2B → OpenRouter → DeepSpeed → HF upload: ConvForever-<depth>-<step>

import argparse
import logging
import os
import tempfile
import shutil
import time
import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import requests
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
import deepspeed
from huggingface_hub import HfApi, ModelCard, ModelCardData

# --- Category constants ---
CATEGORIES = [
    "fish", "reptile", "amphibian", "bird", "mammal", "insect", "arachnid", "crustacean", "mollusk",
    "fungus", "flower", "fruit", "vegetable", "dish", "beverage", "bread", "meat", "dessert",
    "dog", "cat", "primate", "marsupial", "cetacean", "ungulate", "rodent", "raptor", "waterfowl",
    "songbird", "weapon", "tool", "vehicle", "watercraft", "aircraft", "instrument", "electronics",
    "furniture", "apparel", "footwear", "headwear", "accessory", "building", "structure", "fence",
    "bridge", "vessel", "container", "kitchenware", "sports", "geology", "landscape"
]
CATEGORY_SET = set(CATEGORIES)

PROMPT_TEMPLATE = (
    "Summarize this visual caption in one of the categories that describes the image best using this set of words.\n"
    "{categories}\n\n"
    "Caption: \"{caption}\"\n\n"
    "Respond using only one word."
)

# --- Custom ConvNeXt Builder (no timm presets) ---
def make_convnext_by_depth(depth: int, num_classes=1000, in_chans=3, **kwargs):
    """
    Build a ConvNeXt model with exactly `depth` transformer blocks.
    Distributes blocks across 4 stages as evenly as possible (ConvNeXt-style).
    Uses standard ConvNeXt config: dims=[96, 192, 384, 768] for base-scale.
    """
    if depth < 4:
        raise ValueError("Depth must be at least 4 (1 block per stage).")
    
    # Evenly distribute depth across 4 stages
    base = depth // 4
    extra = depth % 4
    depths = [base] * 4
    for i in range(extra):
        depths[i] += 1  # add remainder to early stages

    from timm.models.convnext import ConvNeXt
    # Use standard base dims; you can scale them if desired
    dims = [96, 192, 384, 768]  # This is ConvNeXt-Tiny/Small scale
    model = ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        drop_path_rate=kwargs.get("drop_path_rate", 0.1),
        **{k: v for k, v in kwargs.items() if k != "drop_path_rate"}
    )
    return model, depth

# --- OpenRouter ---
def get_openrouter_response(prompt: str, api_key: str, max_retries=3) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "allenai/olmo-3-1025-7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 1.0
    }
    for attempt in range(max_retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip().lower().rstrip(".")
        except Exception as e:
            logging.warning(f"OpenRouter error (attempt {attempt+1}): {e}")
        time.sleep(1 << attempt)
    return None

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

class TempImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_id = {cat: i for i, cat in enumerate(CATEGORIES)}
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.label_to_id[self.labels[idx]]

def get_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def upload_to_hf(model, step, depth_num, org="collegeofthedesert"):
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

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True, help="Total number of ConvNeXt blocks (e.g., 48)")
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_examples", type=int, default=5000)
    parser.add_argument("--upload_every", type=int, default=500)
    parser.add_argument("--deepspeed", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY")

    # Build model with EXACT depth
    model, actual_depth = make_convnext_by_depth(args.depth, num_classes=len(CATEGORIES), drop_path_rate=0.1)
    logger.info(f"✅ Built ConvNeXt with exactly {actual_depth} layers")

    # DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=args.deepspeed
    )
    device = model_engine.device

    # Stream data
    dataset = load_dataset("laion/relaion2B-en-research", split="train", streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)
    transform = get_transforms()

    total_processed = 0
    global_accepted = 0

    while total_processed < args.max_examples:
        image_paths = []
        labels = []
        temp_dirs = []

        eff_bs = args.micro_batch_size * args.gradient_accumulation_steps
        while len(image_paths) < eff_bs and total_processed < args.max_examples:
            try:
                item = next(dataset_iter)
            except StopIteration:
                break
            total_processed += 1
            caption = item.get("caption", "").strip()
            url = item.get("url", "")
            if not caption or not url:
                continue

            temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
            img_path = download_image_to_path(url, temp_dir)
            if not img_path:
                shutil.rmtree(temp_dir)
                temp_dirs.pop()
                continue

            prompt = PROMPT_TEMPLATE.format(categories=", ".join(sorted(CATEGORY_SET)), caption=caption)
            pred = get_openrouter_response(prompt, api_key)
            if pred not in CATEGORY_SET:
                shutil.rmtree(temp_dir)
                temp_dirs.pop()
                continue

            image_paths.append(img_path)
            labels.append(pred)
            global_accepted += 1

        if not image_paths:
            continue

        # Train
        ds_batch = TempImageDataset(image_paths, labels, transform=transform)
        loader = DataLoader(ds_batch, batch_size=args.micro_batch_size, shuffle=True)
        model_engine.train()
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model_engine(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labs)
            model_engine.backward(loss)
            model_engine.step()

        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

        logger.info(f"Processed: {total_processed}, Accepted: {global_accepted}, Loss: {loss.item():.4f}")

        if global_accepted % args.upload_every == 0:
            base_model = getattr(model_engine, 'module', model_engine)
            upload_to_hf(base_model, global_accepted, actual_depth)

    base_model = getattr(model_engine, 'module', model_engine)
    upload_to_hf(base_model, global_accepted, actual_depth)
    logger.info("✅ Done")

if __name__ == "__main__":
    main()
