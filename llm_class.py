#!/usr/bin/env python
# classify_and_sync.py
# 1. Use Nova 2 Lite (free via OpenRouter) to label captions with closest category
# 2. Save enriched records + sync to Hugging Face Datasets (sharded by attempt)

import argparse
import json
import logging
import hashlib
import os
import re
import time
import requests
from datasets import Dataset
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

def get_image_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]

def query_nova_for_category(caption: str, openrouter_key: str) -> str:
    """Ask Nova 2 Lite to return the best matching category."""
    prompt = (
        f"You are a precise image content classifier. Given the caption: \"{caption}\", "
        f"select the SINGLE most appropriate category from this list:\n[{CATEGORY_LIST_STR}].\n"
        "Respond ONLY with the category name, nothing else."
    )

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": "https://your-site.com",  # optional
        "X-Title": "ConvForever Classifier",       # optional
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

    # Use regex to extract a valid category (case-insensitive, allow quotes/whitespace)
    pattern = r'\b(' + '|'.join(re.escape(c) for c in CATEGORIES) + r')\b'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        logging.warning(f"⚠️ No valid category found in response: '{content}' (caption: {caption[:50]}...)")
        return None

def upload_data_shard(records_slice, start_idx, attempt, org):
    dataset_name = f"ConvForever-data-attempt-{attempt}"
    repo_id = f"{org}/{dataset_name}"

    if not repo_exists(repo_id, repo_type="dataset"):
        create_repo(repo_id, repo_type="dataset", exist_ok=True)

    urls = [r["url"] for r in records_slice]
    captions = [r["caption"] for r in records_slice]
    labels = [r["label"] for r in records_slice]
    image_ids = [get_image_id(url) for url in urls]

    data_dict = {
        "image_id": image_ids,
        "url": urls,
        "caption": captions,
        "label": labels,
        "global_index_start": [start_idx] * len(records_slice),
    }
    dataset = Dataset.from_dict(data_dict)

    try:
        shard_name = f"shard_{start_idx}_{start_idx + len(records_slice)}"
        dataset.push_to_hub(repo_id, config_name=shard_name, split="train", private=False)
        logging.info(f"✅ Uploaded shard '{shard_name}' to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logging.error(f"⚠️ Failed to upload shard: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Classify captions using Nova 2 Lite (free) and sync to HF Datasets.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL with 'url' and 'caption'")
    parser.add_argument("--output_jsonl", type=str, default="classified_captions.jsonl", help="Output labeled JSONL")
    parser.add_argument("--attempt", type=int, required=True, help="Attempt number for dataset versioning")
    parser.add_argument("--org", type=str, default="collegeofthedesert", help="Hugging Face org/username")
    parser.add_argument("--sync_every", type=int, default=500, help="Upload shard every N records")
    parser.add_argument("--openrouter_key", type=str, required=True, help="OpenRouter API key (use free tier)")
    parser.add_argument("--rate_limit_pause", type=float, default=0.5, help="Seconds to wait between API calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load input records
    input_records = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            if line.strip():
                input_records.append(json.loads(line))
    logger.info(f"Loaded {len(input_records)} records from {args.input_jsonl}")

    # Classify each caption
    enriched_records = []
    for i, rec in enumerate(input_records):
        caption = rec.get("caption", "").strip()
        url = rec.get("url", "").strip()
        if not caption or not url:
            logger.warning(f"Skipping record {i}: missing caption or url")
            continue

        label = query_nova_for_category(caption, args.openrouter_key)
        if label is None:
            label = "unknown"  # or skip; here we keep it for audit

        enriched = {
            "url": url,
            "caption": caption,
            "label": label
        }
        enriched_records.append(enriched)

        # Save incrementally to output file
        with open(args.output_jsonl, "a") as out_f:
            out_f.write(json.dumps(enriched) + "\n")

        logger.info(f"[{i+1}/{len(input_records)}] Labeled: {label} <- '{caption[:60]}...'")

        time.sleep(args.rate_limit_pause)  # respect rate limits

    # Sync to HF Datasets
    next_idx = 0
    total = len(enriched_records)
    while next_idx < total:
        end_idx = min(next_idx + args.sync_every, total)
        batch = enriched_records[next_idx:end_idx]
        upload_data_shard(batch, next_idx, args.attempt, args.org)
        next_idx = end_idx

    logger.info(f"✅ Classification and syncing complete for attempt {args.attempt}.")

if __name__ == "__main__":
    main()
