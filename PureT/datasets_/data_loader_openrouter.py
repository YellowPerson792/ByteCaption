"""DataLoader for OpenRouter API models (eval-only, byte streams).
CocoDataset returns corrupted JPEG bytes; collate flattens variants.
"""
from __future__ import annotations

import sys
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch

from lib.config import cfg
from PureT.datasets_.coco_dataset import CocoDataset

def openrouter_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """Validation collate for OpenRouter models.
    Flatten per-sample byte streams and expand metadata to match.
    """
    indices, gv_feat, jpeg_bytes_list = zip(*batch)

    api_image_bytes: List[bytes] = []
    for item in jpeg_bytes_list:
        if isinstance(item, list):
            api_image_bytes.extend(item)
        else:
            api_image_bytes.append(item)

    original_bs = len(jpeg_bytes_list)
    augmentation_factor = len(api_image_bytes) // original_bs if original_bs > 0 else 1

    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)

    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    return expanded_indices, expanded_gv_feat, api_image_bytes, None


def load_val(image_ids_path, gv_feat_path: str = "", max_samples: int = 200):
    corruption_types = list(getattr(cfg.CORRUPTION, "BYTE_STREAM_TYPES", []))
    corruption_level = str(getattr(cfg.CORRUPTION, "BYTE_STREAM_LEVEL", "S0"))
    save_dir = getattr(cfg.INFERENCE, "SAVE_EVAL_IMAGES_DIR", None)
    save_max = int(getattr(cfg.INFERENCE, "SAVE_EVAL_IMAGES_MAX", 0))

    coco_set = CocoDataset(
        image_ids_path=image_ids_path,
        input_seq=None,
        target_seq=None,
        gv_feat_path=gv_feat_path or "",
        seq_per_img=1,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        max_samples=max_samples,
        jpeg_quality=60,
        corruption_types=corruption_types,
        corruption_level=corruption_level,
        model_type="bytecaption",  # need byte stream
        is_training=False,
        save_eval_images_dir=save_dir,
        save_eval_images_max=save_max,
    )

    num_workers = cfg.DATA_LOADER.NUM_WORKERS
    if sys.platform.startswith("win"):
        num_workers = 0
    num_workers = max(0, int(num_workers))
    persistent_workers = bool(getattr(cfg.DATA_LOADER, "PERSISTENT_WORKERS", True)) and num_workers > 0 and not sys.platform.startswith("win")
    pin_memory = bool(getattr(cfg.DATA_LOADER, "PIN_MEMORY", False)) and torch.cuda.is_available()
    prefetch_factor = int(getattr(cfg.DATA_LOADER, "PREFETCH_FACTOR", 2))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=openrouter_collate_val,
        worker_init_fn=None,
        drop_last=False,
        **loader_kwargs,
    )
    return loader
