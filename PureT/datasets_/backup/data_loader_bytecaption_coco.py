"""ByteFormer 版本 COCO 数据加载封装 - 职责明确版本。

数据处理职责分离：
重构后职责：
1. CocoDataset: 内部处理损坏，根据模型类型返回合适格式
    - ByteCaption: 返回损坏后的字节流
    - 其他模型: 返回损坏后解码的PIL图像
2. DataLoader collate: 只负责batch组装和格式转换
    - ByteCaption: bytes -> int32 tensor -> padding
    - 其他模型: 直接使用PIL图像
"""

from __future__ import annotations

import os
import sys
from typing import Any, Sequence, Tuple

import torch
import numpy as np

from lib.config import cfg
from PureT.datasets_.coco_dataset import CocoDataset
import PureT.samplers.distributed as distributed_samplers
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from PureT.byteformer_immigration import get_opts
from PureT.datasets_.data_loader_openrouter import openrouter_collate_val

# 全局变量
_BYTE_STREAM_LENGTHS = []
successful_corruption_samples_saved = 0
SAMPLE_SAVE_DIR = './evaluation_samples'

opts = get_opts()

# 确保 opts 中包含必要的默认值
import argparse as _argparse
if not hasattr(opts, "image_augmentation"):
    opts.image_augmentation = _argparse.Namespace()
if not hasattr(opts.image_augmentation, "pil_save"):
    opts.image_augmentation.pil_save = _argparse.Namespace()
if not hasattr(opts.image_augmentation.pil_save, "corrupt_level"):
    opts.image_augmentation.pil_save.corrupt_level = "none"
if not hasattr(opts.image_augmentation.pil_save, "quality"):
    opts.image_augmentation.pil_save.quality = 60
if not hasattr(opts.image_augmentation, "byte_stream_corrupter"):
    opts.image_augmentation.byte_stream_corrupter = _argparse.Namespace()


def _apply_corruption_cfg_overrides():
    """将 cfg.CORRUPTION 的配置同步到 CoreNet opts。"""
    bs_cfg = getattr(cfg, "CORRUPTION", None)
    if bs_cfg is None:
        return
    bs_opts = opts.image_augmentation.byte_stream_corrupter
    
    # types and level
    if hasattr(bs_cfg, "BYTE_STREAM_TYPES") and bs_cfg.BYTE_STREAM_TYPES is not None:
        bs_opts.types = list(bs_cfg.BYTE_STREAM_TYPES)
    if hasattr(bs_cfg, "BYTE_STREAM_LEVEL") and bs_cfg.BYTE_STREAM_LEVEL:
        bs_opts.level = bs_cfg.BYTE_STREAM_LEVEL
    setattr(opts, "image_augmentation.byte_stream_corrupter.types", getattr(bs_opts, "types", []))
    setattr(opts, "image_augmentation.byte_stream_corrupter.level", getattr(bs_opts, "level", "S0"))
    
    # overrides for RBBF/RBSL/Metadata
    def _assign_if_valid(namespace, key, value):
        if value is None:
            return
        if isinstance(value, (int, float)) and value < 0:
            return
        setattr(namespace, key, value)

    if not hasattr(bs_opts, "rbbf"):
        bs_opts.rbbf = _argparse.Namespace()
    if not hasattr(bs_opts, "rbsl"):
        bs_opts.rbsl = _argparse.Namespace()
    if not hasattr(bs_opts, "metadata_loss"):
        bs_opts.metadata_loss = _argparse.Namespace()

    if hasattr(bs_cfg, "RBBF"):
        _assign_if_valid(bs_opts.rbbf, "trigger_prob", bs_cfg.RBBF.TRIGGER_PROB)
        _assign_if_valid(bs_opts.rbbf, "burst_lambda", bs_cfg.RBBF.BURST_LAMBDA)
        _assign_if_valid(bs_opts.rbbf, "bit_error_rate", bs_cfg.RBBF.BIT_ERROR_RATE)
        setattr(opts, "image_augmentation.byte_stream_corrupter.rbbf.trigger_prob", getattr(bs_opts.rbbf, "trigger_prob", None))
        setattr(opts, "image_augmentation.byte_stream_corrupter.rbbf.burst_lambda", getattr(bs_opts.rbbf, "burst_lambda", None))
        setattr(opts, "image_augmentation.byte_stream_corrupter.rbbf.bit_error_rate", getattr(bs_opts.rbbf, "bit_error_rate", None))
    if hasattr(bs_cfg, "RBSL"):
        _assign_if_valid(bs_opts.rbsl, "trigger_prob", bs_cfg.RBSL.TRIGGER_PROB)
        _assign_if_valid(bs_opts.rbsl, "burst_lambda", bs_cfg.RBSL.BURST_LAMBDA)
        _assign_if_valid(bs_opts.rbsl, "max_drop_ratio", bs_cfg.RBSL.MAX_DROP_RATIO)
        setattr(opts, "image_augmentation.byte_stream_corrupter.rbsl.trigger_prob", getattr(bs_opts.rbsl, "trigger_prob", None))
        setattr(opts, "image_augmentation.byte_stream_corrupter.rbsl.burst_lambda", getattr(bs_opts.rbsl, "burst_lambda", None))
        setattr(opts, "image_augmentation.byte_stream_corrupter.rbsl.max_drop_ratio", getattr(bs_opts.rbsl, "max_drop_ratio", None))
    if hasattr(bs_cfg, "METADATA"):
        _assign_if_valid(bs_opts.metadata_loss, "strip_app_segments", bs_cfg.METADATA.STRIP_APP_SEGMENTS)
        _assign_if_valid(bs_opts.metadata_loss, "zero_prefix_bytes", bs_cfg.METADATA.ZERO_PREFIX_BYTES)
        _assign_if_valid(bs_opts.metadata_loss, "body_trim_ratio", bs_cfg.METADATA.BODY_TRIM_RATIO)
        setattr(opts, "image_augmentation.byte_stream_corrupter.metadata_loss.strip_app_segments", getattr(bs_opts.metadata_loss, "strip_app_segments", None))
        setattr(opts, "image_augmentation.byte_stream_corrupter.metadata_loss.zero_prefix_bytes", getattr(bs_opts.metadata_loss, "zero_prefix_bytes", None))
        setattr(opts, "image_augmentation.byte_stream_corrupter.metadata_loss.body_trim_ratio", getattr(bs_opts.metadata_loss, "body_trim_ratio", None))


# ============================================================================
# Collate 函数 - 统一处理 JPEG 字节流
# ============================================================================

def bytecaption_collate(batch: Sequence[Tuple[Any, ...]]):
    """ByteCaption训练阶段 collate。
    CocoDataset已处理损坏，这里只做格式转换。
    """
    indices, input_seq, target_seq, gv_feat, jpeg_bytes_list = zip(*batch)
    
    original_bs = len(jpeg_bytes_list)
    indices_np = np.stack(indices, axis=0).reshape(-1)
    input_seq_tensor = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq_tensor = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    # CocoDataset已返回处理好的字节流，直接转为int32 tensor
    int32_samples = []
    for jpeg_bytes in jpeg_bytes_list:
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sample_tensor = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
        int32_samples.append({"samples": sample_tensor, "targets": torch.tensor(0)})
    
    # 使用ByteFormer的collate函数进行padding等处理
    collated = byteformer_image_collate_fn(int32_samples, opts)
    att_feats = collated["samples"]
    att_mask = None

    # 如果有样本增强，需要复制元数据
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
    """ByteCaption评估阶段 collate。
    CocoDataset已处理损坏并可能返回多个版本。
    """
    indices, gv_feat, jpeg_bytes_list = zip(*batch)
    
    # 处理可能的多版本损坏结果
    int32_samples = []
    for jpeg_bytes in jpeg_bytes_list:
        # jpeg_bytes可能是单个bytes或bytes列表
        if isinstance(jpeg_bytes, list):
            bytes_list = jpeg_bytes
        else:
            bytes_list = [jpeg_bytes]
        
        for byte_stream in bytes_list:
            _BYTE_STREAM_LENGTHS.append(len(byte_stream))
            buf = np.frombuffer(byte_stream, dtype=np.uint8)
            sample_tensor = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
            int32_samples.append({"samples": sample_tensor, "targets": torch.tensor(0)})
    
    # 使用ByteFormer的collate函数进行padding等处理
    collated = byteformer_image_collate_fn(int32_samples, opts)
    augmented_att_feats = collated["samples"]
    
    # 计算增强因子并复制元数据
    original_bs = len(jpeg_bytes_list)
    if original_bs == 0:
        return torch.tensor(indices), torch.tensor(gv_feat), augmented_att_feats, None

    augmentation_factor = augmented_att_feats.size(0) // original_bs
    
    if augmentation_factor <= 1:
        indices = np.stack(indices, axis=0).reshape(-1)
        gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
        att_mask = None
        return indices, gv_feat, augmented_att_feats, att_mask

    # 复制元数据以匹配增强后的样本数
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    att_mask = None
    return expanded_indices, expanded_gv_feat, augmented_att_feats, att_mask


def visual_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """视觉模型评估collate（已解码PIL）。
    CocoDataset已解码为PIL图像并可能返回多版本。
    """
    global successful_corruption_samples_saved

    indices, gv_feat, pil_images_list = zip(*batch)
    
    # CocoDataset已解码为PIL图像，直接展平
    all_images = []
    for img_or_list in pil_images_list:
        if isinstance(img_or_list, list):
            all_images.extend(img_or_list)
        else:
            all_images.append(img_or_list)
    
    # 计算增强因子
    original_bs = len(pil_images_list)
    augmentation_factor = len(all_images) // original_bs if original_bs > 0 else 1
    
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    return expanded_indices, expanded_gv_feat, all_images, None




# ============================================================================
# DataLoader 构建函数
# ============================================================================

def _worker_init_fn(worker_id: int) -> None:
    """为每个 DataLoader worker 设置独立但可复现的随机种子。"""
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)
    import random as _random
    _random.seed(base_seed + worker_id)


def load_train(distributed: bool, epoch: int, coco_set: CocoDataset):
    """构建训练 DataLoader。"""
    _apply_corruption_cfg_overrides()
    
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
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=bytecaption_collate,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        **loader_kwargs,
    )
    return loader


def load_val(image_ids_path, gv_feat_path: str = '', att_feats_folder=None, max_samples: int = 200, eval_mode='byteformer'):
    """构建验证 DataLoader - CocoDataset内部处理损坏。"""
    _apply_corruption_cfg_overrides()
    
    # 判断使用哪种collate模式
    force_blip = os.getenv("FORCE_BLIP", "").lower() in ("1", "true", "yes")
    force_openrouter = os.getenv("FORCE_OPENROUTER", "").lower() in ("1", "true", "yes")
    model_type = str(getattr(cfg.MODEL, "TYPE", "")).lower()
    is_openrouter = "openrouter" in model_type or model_type.startswith("gpt") or "gpt" in model_type
    is_hf = (
        model_type.startswith("hf")
        or "blip" in model_type
        or "git" in model_type
        or "qwen" in model_type
        or "internvl" in model_type
        or "glm" in model_type
        or "mistral" in model_type
        or "ministral" in model_type
    )

    # 确定模型类型
    if is_hf or force_blip:
        dataset_model_type = "visual"
    else:
        dataset_model_type = "bytecaption"
    
    # 获取损坏配置
    corruption_types = list(getattr(cfg.CORRUPTION, "BYTE_STREAM_TYPES", []))
    corruption_level = str(getattr(cfg.CORRUPTION, "BYTE_STREAM_LEVEL", "S0"))
    
    # 创建CocoDataset - 内部处理损坏和解码
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
        model_type=dataset_model_type,
        is_training=False,
    )
    
    print(f"[数据加载器] 模型类型: {dataset_model_type}")
    print(f"[数据加载器] 损坏配置: {corruption_types} @ {corruption_level}")

    # 选择collate函数
    print(f"[数据加载器] cfg.MODEL.TYPE = {cfg.MODEL.TYPE}")
    if force_openrouter or is_openrouter:
        if force_openrouter and not is_openrouter:
            cfg.MODEL.TYPE = "OPENROUTER"
            print("[data_loader] FORCE_OPENROUTER=1 -> overriding cfg.MODEL.TYPE to OPENROUTER.")
        active_collate_fn = openrouter_collate_val
        print("[data_loader] OpenRouter API: CocoDataset已处理并返回字节流")
    elif force_blip or is_hf:
        if force_blip and "blip" not in model_type:
            print("[数据加载器] FORCE_BLIP=1 -> overriding cfg.MODEL.TYPE to BLIP for collate/eval.")
            cfg.MODEL.TYPE = "BLIP"
        active_collate_fn = visual_collate_val
        print(f"[数据加载器] HF/BLIP/GIT: CocoDataset已解码为PIL")
    else:
        active_collate_fn = bytecaption_collate_val
        print(f"[数据加载器] ByteCaption: CocoDataset已损坏字节流")


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
        collate_fn=active_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        **loader_kwargs,
    )
    return loader
