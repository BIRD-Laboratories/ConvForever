#!/usr/bin/env python
# sync_classified_data.py
# Standalone utility to incrementally sync classified image records to Hugging Face Datasets.
# No training. Just data versioning with image_id and organization control.

import argparse
import json
import logging
import os
import hashlib
import tempfile
from datasets import Dataset
from huggingface_hub import create_repo

# --- Category validation (optional but good for consistency) ---
CATEGORIES = [
    "fish", "reptile", "amphibian", "bird", "mammal", "insect", "arachnid", "crustacean", "mollusk",
    "fungus", "flower", "fruit", "vegetable", "dish", "beverage", "bread", "meat", "dessert",
    "dog", "cat", "primate", "marsupial", "cetacean", "ungulate", "rodent", "raptor", "waterfowl",
    "songbird", "weapon", "tool", "vehicle", "watercraft", "aircraft", "instrument", "electronics",
    "furniture", "apparel", "footwear", "headwear", "accessory", "building", "structure", "fence",
    "bridge", "vessel", "container", "kitchenware", "sports", "geology", "landscape"
]
CATEGORY_SET = set(CATEGORIES)

def get_image_id(url: str) -> str:
    """Generate a short, stable ID from the URL."""
    return hashlib.sha1(url.encode()).hexdigest()[:16]

def upload_data_shard(records_slice, start_idx, depth, org):
    dataset_name = f"ConvForever-data-{depth}"
    repo_id = f"{org}/{dataset_name}"

    urls = [r["url"] for r in records_slice]
    labels = [r["label"] for r in records_slice]
    image_ids = [get_image_id(url) for url in urls]

    # Validate labels (optional safety)
    for lbl in labels:
        if lbl not in CATEGORY_SET:
            raise ValueError(f"Invalid label: {lbl}")

    data_dict = {
        "image_id": image_ids,
        "url": urls,
        "label": labels,
        "global_index_start": [start_idx] * len(records_slice),
    }
    dataset = Dataset.from_dict(data_dict)

    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        shard_name = f"shard_{start_idx}_{start_idx + len(records_slice)}"
        dataset.push_to_hub(repo_id, config_name=shard_name, split="train", private=False)
        logging.info(f"✅ Uploaded shard '{shard_name}' to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logging.error(f"⚠️ Upload failed for shard {shard_name}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Sync classified image records to Hugging Face Datasets.")
    parser.add_argument("--classified_json", type=str, required=True, help="Path to classified JSONL file")
    parser.add_argument("--depth", type=int, required=True, help="Model depth (used in dataset name)")
    parser.add_argument("--sync_every", type=int, default=1000, help="Upload a shard every N records")
    parser.add_argument("--org", type=str, default="collegeofthedesert", help="Hugging Face organization or username")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load records
    records = []
    with open(args.classified_json, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {args.classified_json}")

    # Process in chunks
    next_idx = 0
    total = len(records)
    batch_size = args.sync_every

    while next_idx < total:
        end_idx = min(next_idx + batch_size, total)
        batch = records[next_idx:end_idx]
        upload_data_shard(batch, next_idx, args.depth, args.org)
        next_idx = end_idx

    logger.info("✅ All data shards synced successfully.")

if __name__ == "__main__":
    main()
