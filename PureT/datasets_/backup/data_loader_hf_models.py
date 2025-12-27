"""HF Caption 模型专用数据加载器。

当前约定：
1. CocoDataset 内部处理损坏与解码，HF侧直接接收 PIL 图像
2. 训练/评估的collate只做批次聚合，不再解码或损坏
"""

import os
import sys
from typing import Any, List, Sequence, Tuple

import torch
import numpy as np

from lib.config import cfg
from PureT.datasets_.coco_dataset import CocoDataset
import PureT.samplers.distributed as distributed_samplers


def hf_train_collate_fn(batch: Sequence[Tuple[Any, ...]]):
    """HF模型训练阶段 collate - CocoDataset已返回PIL图像。"""
    indices, captions_list, gv_feat, pil_images_list = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    
    return indices, captions_list, gv_feat, pil_images_list


def hf_val_collate_fn(batch: Sequence[Tuple[Any, ...]]):
    """HF模型评估阶段 collate - CocoDataset已处理损坏和解码。
    
    CocoDataset可能返回：
    - 单个PIL图像（无损坏或单次损坏）
    - PIL图像列表（多次损坏）
    """
    indices, gv_feat, images_data = zip(*batch)
    
    # 展平图像列表
    decoded_images = []
    for img_or_list in images_data:
        if isinstance(img_or_list, list):
            decoded_images.extend(img_or_list)
        else:
            decoded_images.append(img_or_list)
    
    # 同步元数据
    original_bs = len(images_data)
    augmentation_factor = len(decoded_images) // original_bs if original_bs > 0 else 1
    
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)
    
    return expanded_indices, expanded_gv_feat, decoded_images, None


def _worker_init_fn(worker_id: int) -> None:
    """为每个 DataLoader worker 设置独立但可复现的随机种子。"""
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)
    import random as _random
    _random.seed(base_seed + worker_id)


def load_hf_train(distributed: bool, epoch: int, coco_set: CocoDataset):
    """构建HF模型训练 DataLoader。
    
    CocoDataset返回: (indices, captions, gv_feat, pil_images)
    Collate输出: (indices, captions, gv_feat, pil_images)
    """
    # Windows 多进程 DataLoader 支持有限，强制单进程
    num_workers = cfg.DATA_LOADER.NUM_WORKERS
    if sys.platform.startswith("win"):
        num_workers = 0
    num_workers = max(0, int(num_workers))
    persistent_workers = (
        bool(getattr(cfg.DATA_LOADER, "PERSISTENT_WORKERS", True))
        and num_workers > 0
        and not sys.platform.startswith("win")
    )
    pin_memory = bool(getattr(cfg.DATA_LOADER, "PIN_MEMORY", False)) and torch.cuda.is_available()
    prefetch_factor = int(getattr(cfg.DATA_LOADER, "PREFETCH_FACTOR", 2))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    sampler = distributed_samplers.DistributedSampler(coco_set, epoch=epoch) if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    print("[HF训练加载器] CocoDataset已提供PIL图像（训练不损坏）")
    
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=hf_train_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        **loader_kwargs,
    )
    return loader


def load_hf_val(image_ids_path, gv_feat_path: str = '', max_samples: int = 200):
    """构建HF模型验证 DataLoader - CocoDataset内部处理损坏和解码。"""
    # 获取损坏配置
    corruption_types = list(getattr(cfg.CORRUPTION, "BYTE_STREAM_TYPES", []))
    corruption_level = str(getattr(cfg.CORRUPTION, "BYTE_STREAM_LEVEL", "S0"))
    
    # 创建CocoDataset - 内部处理损坏和解码，返回PIL图像
    coco_set = CocoDataset(
        image_ids_path=image_ids_path,
        input_seq=None,  # None 触发 validation mode
        target_seq=None,
        gv_feat_path=gv_feat_path or '',
        seq_per_img=1,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        max_samples=max_samples,
        jpeg_quality=60,
        corruption_types=corruption_types,
        corruption_level=corruption_level,
        model_type="visual",  # HF模型接收PIL图像
        is_training=False,
    )

    print(f"[HF评估加载器] 损坏配置: {corruption_types} @ {corruption_level}")
    print("[HF评估加载器] CocoDataset已处理损坏和解码，返回PIL图像")

    # Windows 下强制单进程 DataLoader
    num_workers = cfg.DATA_LOADER.NUM_WORKERS
    if sys.platform.startswith("win"):
        num_workers = 0
    num_workers = max(0, int(num_workers))
    persistent_workers = (
        bool(getattr(cfg.DATA_LOADER, "PERSISTENT_WORKERS", True))
        and num_workers > 0
        and not sys.platform.startswith("win")
    )
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
        collate_fn=hf_val_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        **loader_kwargs,
    )
    return loader
