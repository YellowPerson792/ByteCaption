"""Flickr8k DataLoader 封装 (ByteFormer 版本)。
"""

from __future__ import annotations

import os
import sys
import torch
import numpy as np
from typing import Any, List, Sequence, Tuple

from lib.config import cfg
from PureT.datasets_.flickr8k_dataset import Flickr8kDataset
import PureT.samplers.distributed as distributed_samplers
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from PureT.byteformer_immigration import get_opts

opts = get_opts()

def byteformer_collate(batch):
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)
    
    original_bs = len(att_feats)
    indices_np = np.stack(indices, axis=0).reshape(-1)
    input_seq_tensor = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq_tensor = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    """
    # 读取图像的预训练特征时，大小为[L, D]，其中L的长度可能不一（如目标特征）
    # 因此需要进行特征数量判断，并生成特征掩码 att_mask
    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)
    """
    # 图像特征，无需与预训练特征一样进行特征数量判断，直接合并即可
    # att_mask为最终grid特征大小，实际上grid特征无需att_mask亦可  
    
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 224, 224]
    
    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
    collated = byteformer_image_collate_fn(corenet_batch, opts)
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

def byteformer_collate_val(batch):
    indices, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    """
    # 读取图像的预训练特征时，大小为[L, D]，其中L的长度可能不一（如目标特征）
    # 因此需要进行特征数量判断，并生成特征掩码 att_mask
    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)
    """
    # 图像特征，无需与预训练特征一样进行特征数量判断，直接合并即可
    # att_mask为最终grid特征大小，实际上grid特征无需att_mask亦可
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 224, 224]
    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    att_feats = collated["samples"]
    att_mask = None

    return indices, gv_feat, att_feats, att_mask


def _worker_init_fn(worker_id: int):
    """为每个 worker 设置确定性随机种子 (兼容 numpy & torch)。"""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)


def load_train(distributed, epoch, flickr_set):
    sampler = (
        distributed_samplers.DistributedSampler(flickr_set, epoch=epoch)
        if distributed
        else None
    )
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False

    # Windows 多进程 DataLoader 支持有限，强制单进程避免 spawn 造成卡死/日志噪声
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
        flickr_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=byteformer_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn,
        **loader_kwargs,
    )
    return loader

def load_val(image_ids_path, gv_feat_path='', att_feats_folder=None, max_samples=200):
    # 直接将 None 传递给数据集以触发验证模式 (input_seq=None & target_seq=None)
    flickr_set = Flickr8kDataset(
        image_ids_path=image_ids_path,
        input_seq=None,
        target_seq=None,
        gv_feat_path=gv_feat_path or '',
        seq_per_img=1,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        max_samples=max_samples,
    )

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
        flickr_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=byteformer_collate_val,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn,
        **loader_kwargs,
    )
    return loader
