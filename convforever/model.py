"""
Model creation utilities for ConvForever
"""

import torch.nn as nn


# --- Category constants (must match classify_captions.py) ---
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
    """Create a ConvNeXt model with a specific depth."""
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