#!/usr/bin/env python
"""
Import LAION dataset from JSONL file.
This script processes the JSONL file with image URLs and labels to prepare for training.
"""

import json
import argparse
import logging
from convforever.dataset import JsonImageDataset


def main():
    parser = argparse.ArgumentParser(description="Import LAION dataset from JSONL file")
    parser.add_argument("--input_jsonl", type=str, required=True, 
                        help="Path to input JSONL file with image URLs and labels")
    parser.add_argument("--output_records", type=str, default="laion_records.json",
                        help="Path to save processed records")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to process (for testing)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load records from JSONL
    records = []
    with open(args.input_jsonl, "r") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if line.strip():
                record = json.loads(line)
                records.append(record)
    
    logger.info(f"Loaded {len(records)} records from {args.input_jsonl}")
    
    # Validate records format
    if records:
        sample_record = records[0]
        if "url" not in sample_record or "label" not in sample_record:
            raise ValueError("Each record must contain 'url' and 'label' fields")
    
    # Save records to file
    with open(args.output_records, "w") as f:
        json.dump(records, f)
    
    logger.info(f"Saved {len(records)} records to {args.output_records}")
    
    # Test dataset creation
    from convforever.model import CATEGORIES
    label_to_id = {cat: i for i, cat in enumerate(CATEGORIES)}
    
    valid_records = []
    for record in records:
        if record["label"] in label_to_id:
            valid_records.append(record)
    
    logger.info(f"Found {len(valid_records)} records with valid labels out of {len(records)} total")
    

if __name__ == "__main__":
    main()