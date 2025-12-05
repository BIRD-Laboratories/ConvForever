#!/usr/bin/env python
# classify_captions.py
# Stream relaion2B captions, classify with OpenRouter, save valid (ID, caption, URL, label) to JSON.

import argparse
import json
import logging
import os
import time
import requests
from datasets import load_dataset

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

def get_openrouter_response(prompt: str, api_key: str, max_retries=3) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "amazon/nova-2-lite-v1:free",
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=5000)
    parser.add_argument("--output_json", type=str, default="classified_captions.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY")

    dataset = load_dataset("laion/relaion2B-en-research", split="train", streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)

    total_processed = 0
    accepted_count = 0

    with open(args.output_json, "w") as f_out:
        while total_processed < args.max_examples:
            try:
                item = next(dataset_iter)
            except StopIteration:
                break

            total_processed += 1
            caption = item.get("caption", "").strip()
            url = item.get("url", "")
            sample_id = item.get("uid") or f"item_{total_processed}"

            if not caption or not url:
                continue

            prompt = PROMPT_TEMPLATE.format(categories=", ".join(sorted(CATEGORY_SET)), caption=caption)
            pred = get_openrouter_response(prompt, api_key)

            if pred in CATEGORY_SET:
                record = {
                    "id": sample_id,
                    "caption": caption,
                    "url": url,
                    "label": pred
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
                accepted_count += 1
                logger.info(f"Accepted {accepted_count}: {pred} ← {caption[:60]}...")

            if total_processed % 100 == 0:
                logger.info(f"Processed: {total_processed}, Accepted: {accepted_count}")

    logger.info(f"✅ Done. Saved {accepted_count} labeled captions to {args.output_json}")

if __name__ == "__main__":
    main()
