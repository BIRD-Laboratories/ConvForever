#!/usr/bin/env python
"""
Estimate parameter count and VRAM usage for a custom ConvNeXt model.
Supports arbitrary depth via stage-wise block distribution.
"""

import argparse
import math

def calculate_convnext_params(depths, dims, in_chans=3, num_classes=1000, patch_size=4):
    """
    Approximate parameter count for ConvNeXt (based on timm implementation).
    - Each block: 2 conv layers (1x1 + DW 7x7), LayerNorm, MLP
    - Stage 0 has stem (patchify)
    """
    params = 0

    # Stem: patchify with conv (H/4, W/4)
    params += in_chans * dims[0] * (patch_size ** 2)

    # Stages
    for i, (depth, dim) in enumerate(zip(depths, dims)):
        # Each block:
        # - 1x1 conv (dim → dim)
        # - DW conv (7x7, dim channels)
        # - 2x LayerNorm (no params)
        # - MLP (dim → 4*dim → dim)
        block_params = (
            (dim * dim) +                     # 1x1 conv
            (dim * 7 * 7) +                   # DW conv
            (4 * dim * dim) +                 # MLP (linear + gelu + linear)
            (dim) + (4 * dim) + (dim)         # biases (approx)
        )
        params += depth * block_params

        # Downsampling (except last stage): LN + 1x1 conv
        if i < len(dims) - 1:
            next_dim = dims[i + 1]
            params += dim  # LayerNorm bias approx
            params += dim * next_dim  # 1x1 conv weight
            params += next_dim  # bias

    # Final head: GAP + classifier
    params += dims[-1] * num_classes  # classifier weight
    params += num_classes  # bias

    return params

def format_num(x):
    if x < 1e6:
        return f"{x/1e3:.1f}K"
    elif x < 1e9:
        return f"{x/1e6:.1f}M"
    else:
        return f"{x/1e9:.1f}B"

def estimate_vram_gb(
    param_count,
    batch_size,
    img_size=224,
    bytes_per_param=2,    # fp16
    zero_stage=2,
    offload_optimizer=False
):
    """
    Rough VRAM estimate (GB) for training.
    Based on: https://deepspeed.ai/tutorials/zero/
    """
    # Model states (fp16): params + grads + optimizer states (Adam: 2x fp32)
    if zero_stage >= 2:
        model_states_bytes = param_count * (2 + 0 + (0 if offload_optimizer else 8))
    elif zero_stage == 1:
        model_states_bytes = param_count * (2 + 2 + (8 if not offload_optimizer else 0))
    else:
        model_states_bytes = param_count * (2 + 2 + 8)

    # Activations (rough): batch_size * img_size^2 * num_channels * depth * 4 bytes
    # Very rough heuristic: ~20 bytes per pixel per layer
    activation_bytes = batch_size * (img_size ** 2) * 20 * sum([d for d in [30] for _ in range(1)])  # placeholder
    # Better: skip detailed activation calc; focus on model states (dominant for large models)

    total_bytes = model_states_bytes + activation_bytes
    return total_bytes / (1024**3)

def distribute_depth_to_stages(total_depth, num_stages=4):
    """Evenly distribute total depth across stages (like ConvNeXt)."""
    base = total_depth // num_stages
    extra = total_depth % num_stages
    depths = [base] * num_stages
    for i in range(extra):
        depths[i] += 1
    return depths

def main():
    parser = argparse.ArgumentParser(description="Estimate ConvNeXt params and VRAM")
    parser.add_argument("--depth", type=int, required=True, help="Total number of blocks (e.g., 48)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num_classes", type=int, default=58, help="Number of output classes")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0,1,2,3])
    parser.add_argument("--offload_optimizer", action="store_true")
    args = parser.parse_args()

    # Standard ConvNeXt dims (Tiny/Small scale)
    dims = [96, 192, 384, 768]
    depths = distribute_depth_to_stages(args.depth)

    param_count = calculate_convnext_params(
        depths=depths,
        dims=dims,
        num_classes=args.num_classes
    )

    vram_gb = estimate_vram_gb(
        param_count=param_count,
        batch_size=args.batch_size,
        zero_stage=args.zero_stage,
        offload_optimizer=args.offload_optimizer
    )

    print("\n" + "="*50)
    print(f"ConvNeXt Configuration")
    print("="*50)
    print(f"Total depth       : {args.depth}")
    print(f"Stage depths      : {depths}")
    print(f"Channel dims      : {dims}")
    print(f"Num classes       : {args.num_classes}")
    print()
    print(f"Estimated params  : {format_num(param_count)} ({param_count:,})")
    print(f"VRAM per GPU (fp16 + ZeRO-{args.zero_stage}" +
          (" + offload" if args.offload_optimizer else "") +
          f") : ~{vram_gb:.1f} GB")
    print("="*50)

if __name__ == "__main__":
    main()
