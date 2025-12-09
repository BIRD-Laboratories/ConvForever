#!/usr/bin/env python
# classify_and_sync.py
# Stream from Hugging Face dataset, classify captions with Nova 2 Lite, sync shards to HF

import argparse
import json
import logging
import hashlib
import os
import re
import time
import requests
from datasets import load_dataset
from huggingface_hub import create_repo, repo_exists

# --- Known categories ---
CATEGORIES = [
    "fish", "reptile", "amphibian", "bird", "mammal", "insect", "arachnid", "crustacean", "mollusk",
    "fungus", "flower", "fruit", "vegetable", "dish", "beverage", "bread", "meat", "dessert",
    "dog", "cat", "primate", "marsupial", "cetacean", "ungulate", "rodent", "raptor", "waterfowl",
    "songbird", "weapon", "tool", "vehicle", "watercraft", "aircraft", "instrument", "electronics",
    "furniture", "apparel", "footwear", "headwear", "accessory", "building", "structure", "fence",
    "bridge", "vessel", "container", "kitchenware", "sports", "geology", "landscape"
]
CATEGORY_SET = set(CATEGORIES)
CATEGORY_LIST_STR = ", ".join(f'"{c}"' for c in CATEGORIES)

def get_image_id(identifier: str) -> str:
    return hashlib.sha1(str(identifier).encode()).hexdigest()[:16]

def query_nova_for_category(caption: str, openrouter_key: str) -> str:
    prompt = (
        f"You are a precise image content classifier. Given the caption: \"{caption}\", "
        f"select the SINGLE most appropriate category from this list:\n[{CATEGORY_LIST_STR}].\n"
        "Respond ONLY with the category name, nothing else."
    )

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "ConvForever Classifier",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "amazon/nova-2-lite-v1:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"❌ API error for caption '{caption[:50]}...': {e}")
        return None

    pattern = r'\b(' + '|'.join(re.escape(c) for c in CATEGORIES) + r')\b'
    match = re.search(pattern, content, re.IGNORECASE)
    return match.group(1).lower() if match else None

def upload_data_shard(records_slice, start_idx, attempt, org):
    dataset_name = f"ConvForever-data-attempt-{attempt}"
    repo_id = f"{org}/{dataset_name}"

    if not repo_exists(repo_id, repo_type="dataset"):
        create_repo(repo_id, repo_type="dataset", exist_ok=True)

    ids = [r["id"] for r in records_slice]
    captions = [r["caption"] for r in records_slice]
    labels = [r["label"] for r in records_slice]
    image_ids = [get_image_id(i) for i in ids]

    data_dict = {
        "image_id": image_ids,
        "id": ids,
        "caption": captions,
        "label": labels,
        "global_index_start": [start_idx] * len(records_slice),
    }

    from datasets import Dataset
    dataset = Dataset.from_dict(data_dict)
    shard_name = f"shard_{start_idx}_{start_idx + len(records_slice)}"
    dataset.push_to_hub(repo_id, config_name=shard_name, split="train", private=False)
    logging.info(f"✅ Uploaded shard '{shard_name}' to https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Classify captions from Hugging Face dataset using Nova 2 Lite.")
    parser.add_argument("--dataset", type=str, default="Spawning/pd-extended",
                        help="Hugging Face dataset ID to stream from")
    parser.add_argument("--caption_field", type=str, default="caption",
                        help="Field name for caption in dataset")
    parser.add_argument("--id_field", type=str, default="id",
                        help="Field name for unique ID in dataset")
    parser.add_argument("--attempt", type=int, required=True, help="Attempt number for versioning")
    parser.add_argument("--org", type=str, default="collegeofthedesert", help="Hugging Face org/user")
    parser.add_argument("--sync_every", type=int, default=500, help="Upload shard every N records")
    parser.add_argument("--openrouter_key", type=str, required=True, help="OpenRouter API key")
    parser.add_argument("--rate_limit_pause", type=float, default=0.5, help="Pause between API calls (sec)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Stream and shuffle dataset
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=args.seed)

    enriched_records = []
    output_path = f"classified_attempt_{args.attempt}.jsonl"

    for i, example in enumerate(shuffled_dataset):
        caption = example.get(args.caption_field, "").strip()
        identifier = example.get(args.id_field)
        if not caption or identifier is None:
            logging.warning(f"Skipping record {i}: missing caption or ID")
            continue

        label = query_nova_for_category(caption, args.openrouter_key)
        if label is None:
            label = "unknown"

        rec = {"id": identifier, "caption": caption, "label": label}
        enriched_records.append(rec)

        # Append to file immediately
        with open(output_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        logging.info(f"[{i+1}] Labeled: {label} <- '{caption[:60]}...'")

        time.sleep(args.rate_limit_pause)

        # Upload shard if needed
        if (i + 1) % args.sync_every == 0:
            upload_data_shard(enriched_records, i + 1 - len(enriched_records), args.attempt, args.org)
            enriched_records = []

    # Upload final partial shard
    if enriched_records:
        upload_data_shard(enriched_records, len(enriched_records), args.attempt, args.org)

    logging.info(f"✅ Completed attempt {args.attempt}. Output saved to {output_path}")

if __name__ == "__main__":
    main()
