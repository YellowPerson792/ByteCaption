"""ByteFormer 版本 COCO 数据加载封装。
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Sequence, Tuple

import torch
import numpy as np
from torchvision import transforms  # noqa: F401 (预留未来扩展)

from PureT.lib.config import cfg
from PureT.datasets_.coco_dataset_hf import CocoDataset
import PureT.samplers.distributed as distributed_samplers
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from PureT.byteformer_immigration import get_opts

# 加了一个BLIP专用的collate函数，导入这些模块，先放到这里
import io
from PIL import Image
from corenet.data.transforms import image_bytes
from torchvision import transforms as T # 使用别名避免冲突

# --- START: 为码流长度统计添加全局变量 ---
_BYTE_STREAM_LENGTHS = []
# --- END: 为码流长度统计添加全局变量 ---

# --- START: 为示例输出添加新依赖和全局变量 ---
import os
from torchvision.transforms.functional import to_pil_image

successful_corruption_samples_saved = 0
SAMPLE_SAVE_DIR = './evaluation_samples'
# --- END: 为示例输出添加新依赖和全局变量 ---

opts = get_opts()

# 确保 opts 中包含 image_augmentation.pil_save.corrupt_level 的默认值，
# 以便上层注入或默认 "none" 时 transform 能安全读取。
import argparse as _argparse
if not hasattr(opts, "image_augmentation"):
    opts.image_augmentation = _argparse.Namespace()
if not hasattr(opts.image_augmentation, "pil_save"):
    opts.image_augmentation.pil_save = _argparse.Namespace()
if not hasattr(opts.image_augmentation.pil_save, "corrupt_level"):
    opts.image_augmentation.pil_save.corrupt_level = "none"

def sample_collate(batch):
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
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
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 384, 384]
    att_mask = torch.ones(att_feats.size()[0], 12*12)

    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask

def sample_collate_val(batch):
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
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 384, 384]
    att_mask = torch.ones(att_feats.size()[0], 12*12)

    return indices, gv_feat, att_feats, att_mask

def byteformer_collate(batch: Sequence[Tuple[Any, ...]]):
    """
    训练阶段 collate。
    """
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
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
    
    att_feats = torch.stack(att_feats, 0)
    
    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    att_feats = collated["samples"]
    att_mask = None
    
    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask

def byteformer_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """验证阶段 collate。
    已修改以支持样本堆叠增强，会同步复制所有元数据。
    """
    indices, gv_feat, att_feats = zip(*batch)
    
    # 1. 将图像张量打包成 corenet 期望的格式
    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target

    # 2. 调用核心 collate 函数，这将执行 1->N 的增强
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    augmented_att_feats = collated["samples"]
    
    # 3. 计算增强因子（例如，4）
    original_bs = len(att_feats)
    if original_bs == 0:
        # 处理空批次的情况
        return torch.tensor(indices), torch.tensor(gv_feat), augmented_att_feats, None

    augmentation_factor = augmented_att_feats.size(0) // original_bs
    
    # 如果没有增强，直接返回
    if augmentation_factor <= 1:
        indices = np.stack(indices, axis=0).reshape(-1)
        gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
        att_mask = None # ByteFormer collate 后通常为 None
        return indices, gv_feat, augmented_att_feats, att_mask

    # 4. 关键修复：将所有元数据复制 N 份以匹配增强后的数据
    print(f"[DEBUG Metadata] Augmentation factor is {augmentation_factor}. Duplicating metadata.")
    
    # 复制 indices
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    # 复制 gv_feat
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    # gv_feat 的形状是 [B, D]，我们需要将其扩展为 [B*N, D]
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    att_mask = None # ByteFormer collate 后通常为 None

    return expanded_indices, expanded_gv_feat, augmented_att_feats, att_mask

def blip_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """
    验证阶段 collate，专门为 BLIP 模型准备数据。
    它会模拟 ByteFormer 的损坏流程，并尝试解码图像。
    """
    # --- START: 引用全局计数器 ---
    global successful_corruption_samples_saved
    # --- END: 引用全局计数器 ---

    indices, gv_feat, att_feats = zip(*batch)
    
    # 1. 初始化 BLIP 的图像处理器和 ByteFormer 的损坏器
    blip_image_tensors = []
    corrupter = image_bytes.ByteStreamCorrupter(opts)

    # 2. 对批次中的每个原始图像执行“编码 -> 损坏 -> 解码”流程
    for i, img_tensor in enumerate(att_feats):
        try:
            byte_stream = image_bytes._image_to_bytes(img_tensor, format="jpeg", quality=95)
            original_bytes = byte_stream.getvalue()

            # --- START: 收集码流长度 ---
            _BYTE_STREAM_LENGTHS.append(len(original_bytes))
            # --- END: 收集码流长度 ---

        except Exception as e:
            print(f"[ERROR] Failed to encode image to bytes: {e}")
            num_corruptions = len(corrupter.corruption_types) if corrupter.corruption_types else 1
            blip_image_tensors.extend([None] * num_corruptions)
            continue

        corruption_types_to_apply = corrupter.corruption_types if corrupter.corruption_types else ["none"]
        for corruption_type in corruption_types_to_apply:
            corrupted_bytes = original_bytes
            if corruption_type != "none":
                if corruption_type == "bit_flip":
                    corrupted_bytes = corrupter._random_bit_flip(original_bytes, corrupter.params["bit_flip"])
                elif corruption_type == "segment_dropout":
                    corrupted_bytes = corrupter._segment_dropout(original_bytes, corrupter.params["drop"])
                elif corruption_type == "header_truncation":
                    corrupted_bytes = corrupter._header_truncation(original_bytes, corrupter.params["head"])
                elif corruption_type == "tail_truncation":
                    corrupted_bytes = corrupter._tail_truncation(original_bytes, corrupter.params["tail"])
            
            try:
                reconstructed_img = Image.open(io.BytesIO(corrupted_bytes)).convert("RGB")
                blip_image_tensors.append(reconstructed_img)

                # --- START: 添加小样本保存逻辑 ---
                if corruption_type != "none" and successful_corruption_samples_saved < 5:
                    # 确保保存目录存在
                    os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)
                    
                    # 创建一个描述性的文件名
                    # 注意：这里我们无法直接获取原始 image_id，所以使用一个全局唯一的样本编号
                    filename = f"decoded_corrupted_sample_{successful_corruption_samples_saved}_{corruption_type}_{corrupter.level}.jpg"
                    filepath = os.path.join(SAMPLE_SAVE_DIR, filename)
                    
                    # 保存图像
                    reconstructed_img.save(filepath)
                    
                    print("\n" + "─" * 70)
                    print(f"[损坏图像保存成功]")
                    print(f"  - 损坏类型: {corruption_type}_{corrupter.level}")
                    print(f"  - 图像已保存至: {filepath}")
                    print("─" * 70)
                    
                    successful_corruption_samples_saved += 1
                # --- END: 添加小样本保存逻辑 ---

            except Exception:
                # 解码失败，用 None 作为占位符
                blip_image_tensors.append(None)

    # 3. 同步元数据，使其数量与增强后的图像数量匹配
    original_bs = len(att_feats)
    augmentation_factor = len(blip_image_tensors) // original_bs if original_bs > 0 else 1
    
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    # 我们将 BLIP 的数据放在原本 att_feats 的位置，以保持返回结构一致
    return expanded_indices, expanded_gv_feat, blip_image_tensors, None


def _worker_init_fn(worker_id: int) -> None:
    """为每个 DataLoader worker 设置独立但可复现的随机种子。"""
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)
    import random as _random
    _random.seed(base_seed + worker_id)


def load_train(distributed: bool, epoch: int, coco_set: CocoDataset):
    """构建训练 DataLoader。

    参数：
        distributed: 是否分布式
        epoch: 当前 epoch (用于分布式 sampler 设置 shuffle seed)
        coco_set: 已实例化的 CocoDataset
    """
    sampler = distributed_samplers.DistributedSampler(coco_set, epoch=epoch) if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if cfg.DATA_LOADER.NUM_WORKERS > 0 else False,
        collate_fn=byteformer_collate,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    return loader

def load_val(image_ids_path, gv_feat_path: str = '', att_feats_folder=None, max_samples: int = 200, eval_mode='byteformer'):  # noqa: D401
    """构建验证 DataLoader（进入数据集 validation 模式）。"""
    coco_set = CocoDataset(
        image_ids_path=image_ids_path,
        input_seq=None,  # None 触发 validation mode
        target_seq=None,
        gv_feat_path=gv_feat_path or '',
        seq_per_img=1,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        max_samples=max_samples,
    )

    # 加一个选择，先这样加，之后再说
    if cfg.MODEL.TYPE == 'BLIP':
        active_collate_fn = blip_collate_val
        print(f"[数据加载器] 已配置为 BLIP 评估模式。")
    elif cfg.MODEL.TYPE == 'PureT_byteformer':
        active_collate_fn = byteformer_collate_val
        print(f"[数据加载器] 已配置为 ByteFormer 评估模式。")
    else:
        raise ValueError(f"未知的模型类型 '{cfg.MODEL.TYPE}' 在配置文件中。")


    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if cfg.DATA_LOADER.NUM_WORKERS > 0 else False,
        collate_fn=active_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    return loader
