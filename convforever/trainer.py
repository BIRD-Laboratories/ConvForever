"""
Training utilities for ConvForever
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deepspeed

from .dataset import JsonImageDataset


def train_with_deepspeed(model, args, device, records, transform):
    """Train model using DeepSpeed."""
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=args.deepspeed_config if hasattr(args, 'deepspeed_config') else args.deepspeed,
        lr=args.lr
    )
    device = model_engine.device
    
    # Train in batches
    eff_bs = args.micro_batch_size * args.gradient_accumulation_steps
    global_step = 0
    
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

        logging.info(f"Batch {i // eff_bs + 1}, Loss: {loss.item():.4f}")

        if global_step % args.upload_every == 0:
            base_model = getattr(model_engine, 'module', model_engine)
            from .utils import upload_to_hf
            upload_to_hf(base_model, global_step, args.depth, args.org)

    return model_engine, global_step


def train_without_deepspeed(model, args, device, records, transform):
    """Train model without DeepSpeed (regular PyTorch training)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train in batches
    eff_bs = args.micro_batch_size * args.gradient_accumulation_steps
    global_step = 0
    
    for i in range(0, len(records), eff_bs):
        batch_records = records[i:i + eff_bs]
        if len(batch_records) == 0:
            continue

        ds_batch = JsonImageDataset(batch_records, transform=transform)
        loader = DataLoader(ds_batch, batch_size=args.micro_batch_size, shuffle=True)

        model.train()
        for imgs, labs in loader:
            if imgs is None or labs is None:
                continue
            imgs, labs = imgs.to(device), labs.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            global_step += 1

        logging.info(f"Batch {i // eff_bs + 1}, Loss: {loss.item():.4f}")

        if global_step % args.upload_every == 0:
            from .utils import upload_to_hf
            upload_to_hf(model, global_step, args.depth, args.org)

    return model, global_step