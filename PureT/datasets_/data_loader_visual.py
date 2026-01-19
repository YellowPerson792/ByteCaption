"""DataLoader for visual models (BLIP/GIT/HF vision) that consume PIL images.
CocoDataset applies corruption and returns decoded PIL images when model_type="visual".
"""
from __future__ import annotations

import sys
from typing import Any, Sequence, Tuple

import numpy as np
import torch

from lib.config import cfg
from PureT.datasets_.coco_dataset import CocoDataset


def visual_collate(batch: Sequence[Tuple[Any, ...]]):
    """Training collate: passthrough PIL images."""
    indices, captions_list, gv_feat, pil_images_list = zip(*batch)
    indices_np = np.stack(indices, axis=0).reshape(-1)
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    return indices_np, captions_list, gv_feat_tensor, list(pil_images_list)


def visual_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """Eval collate: flatten possible augmented PIL variants."""
    indices, gv_feat, pil_images_list = zip(*batch)
    all_images = []
    for img_or_list in pil_images_list:
        if isinstance(img_or_list, list):
            all_images.extend(img_or_list)
        else:
            all_images.append(img_or_list)

    original_bs = len(pil_images_list)
    augmentation_factor = len(all_images) // original_bs if original_bs > 0 else 1

    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)

    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    return expanded_indices, expanded_gv_feat, all_images, None


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
        model_type="visual",
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
        collate_fn=visual_collate_val,
        worker_init_fn=None,
        drop_last=False,
        **loader_kwargs,
    )
    return loader
