"""DataLoader for ByteCaption byte-stream models (train & eval).
CocoDataset handles corruption and returns JPEG bytes; collates convert to int32 tensors.
"""
from __future__ import annotations

import sys
from typing import Any, Sequence, Tuple

import numpy as np
import torch

from lib.config import cfg
from PureT.datasets_.coco_dataset import CocoDataset
import PureT.samplers.distributed as distributed_samplers
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from PureT.byteformer_immigration import get_opts

opts = get_opts()

# ---------------------------------------------------------------------------
# Collates
# ---------------------------------------------------------------------------

def bytecaption_collate(batch: Sequence[Tuple[Any, ...]]):
    """Training collate: bytes -> int32 tensor -> padding."""
    indices, input_seq, target_seq, gv_feat, jpeg_bytes_list = zip(*batch)

    original_bs = len(jpeg_bytes_list)
    indices_np = np.stack(indices, axis=0).reshape(-1)
    input_seq_tensor = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq_tensor = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    int32_samples = []
    for jpeg_bytes in jpeg_bytes_list:
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sample_tensor = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
        int32_samples.append({"samples": sample_tensor, "targets": torch.tensor(0)})

    collated = byteformer_image_collate_fn(int32_samples, opts)
    att_feats = collated["samples"]
    att_mask = None

    augmentation_factor = att_feats.size(0) // original_bs if original_bs > 0 else 1
    if augmentation_factor > 1:
        indices = np.repeat(indices_np, augmentation_factor, axis=0)
        gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)
        seq_per_img = max(int(getattr(cfg.DATA_LOADER, "SEQ_PER_IMG", 1)), 1)
        input_seq = input_seq_tensor.view(original_bs, seq_per_img, -1)
        target_seq = target_seq_tensor.view(original_bs, seq_per_img, -1)
        input_seq = input_seq.repeat_interleave(augmentation_factor, dim=0).view(-1, input_seq_tensor.size(-1))
        target_seq = target_seq.repeat_interleave(augmentation_factor, dim=0).view(-1, target_seq_tensor.size(-1))
    else:
        indices = indices_np
        gv_feat = gv_feat_tensor
        input_seq = input_seq_tensor
        target_seq = target_seq_tensor

    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask


def bytecaption_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """Eval collate: supports multiple corrupted byte variants per sample."""
    indices, gv_feat, jpeg_bytes_list = zip(*batch)

    int32_samples = []
    for jpeg_bytes in jpeg_bytes_list:
        bytes_list = jpeg_bytes if isinstance(jpeg_bytes, list) else [jpeg_bytes]
        for byte_stream in bytes_list:
            buf = np.frombuffer(byte_stream, dtype=np.uint8)
            sample_tensor = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
            int32_samples.append({"samples": sample_tensor, "targets": torch.tensor(0)})

    collated = byteformer_image_collate_fn(int32_samples, opts)
    augmented_att_feats = collated["samples"]

    original_bs = len(jpeg_bytes_list)
    if original_bs == 0:
        return torch.tensor(indices), torch.tensor(gv_feat), augmented_att_feats, None

    augmentation_factor = augmented_att_feats.size(0) // original_bs
    if augmentation_factor <= 1:
        indices = np.stack(indices, axis=0).reshape(-1)
        gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
        att_mask = None
        return indices, gv_feat, augmented_att_feats, att_mask

    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)

    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    att_mask = None
    return expanded_indices, expanded_gv_feat, augmented_att_feats, att_mask


# ---------------------------------------------------------------------------
# Loader builders
# ---------------------------------------------------------------------------


def load_train(distributed: bool, epoch: int, coco_set: CocoDataset):
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

    sampler = distributed_samplers.DistributedSampler(coco_set, epoch=epoch) if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=bytecaption_collate,
        worker_init_fn=None,
        drop_last=False,
        **loader_kwargs,
    )
    return loader


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
        model_type="bytecaption",
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
        collate_fn=bytecaption_collate_val,
        worker_init_fn=None,
        drop_last=False,
        **loader_kwargs,
    )
    return loader
