import os
import sys
import argparse
import io

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
# --- START: 关键修复 - 导入 torchvision.transforms ---
from torchvision import transforms
# --- END: 关键修复 ---

# --- 设置项目路径，以便导入自定义模块 ---
# 假设此脚本位于 PureT 目录下，项目根目录是 ByteCaption
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------

from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.lib.config import cfg, cfg_from_file

def analyze_lengths(args):
    """
    主分析函数：加载数据集，处理图像，并打印统计数据。
    """
    print(f"--- 分析 JPEG 码流长度 ---")
    print(f"数据集: {args.dataset_path}")
    print(f"处理样本数: {'ALL' if args.max_samples <= 0 else args.max_samples}")
    print(f"JPEG 质量: {args.quality}")
    print("-" * 30)

    # 1. 加载数据集
    try:
        coco_set = CocoDataset(
            image_ids_path=args.dataset_path,
            input_seq=None,  # 触发验证模式
            target_seq=None,
            gv_feat_path='',
            seq_per_img=1,
            max_feat_num=-1,
            max_samples=args.max_samples if args.max_samples > 0 else None,
        )
        print(f"成功加载数据集，共 {len(coco_set)} 个样本。")
    except Exception as e:
        print(f"[错误] 加载数据集失败: {e}")
        print("请确保路径正确，并且项目结构允许从 PureT 目录导入 datasets_ 模块。")
        return

    # 2. 创建 DataLoader
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. 循环处理并收集长度
    byte_stream_lengths = []
    pbar = tqdm(loader, desc="处理图像中", unit="batch")

    # --- START: 关键修复 - 定义 224*224 缩放变换 ---
    resize_transform = transforms.Resize((224, 224), antialias=True)
    # --- END: 关键修复 ---

    for batch in pbar:
        # CocoDataset 在验证模式下返回 (indices, gv_feat, att_feats)
        # 我们只需要 att_feats，即图像张量
        _, _, image_tensors = batch

        for img_tensor in image_tensors:
            try:
                # --- START: 关键修复 - 应用缩放变换 ---
                # 确保无论输入张量是什么尺寸，都先将其缩放到 224x224
                resized_tensor = resize_transform(img_tensor)
                # --- END: 关键修复 ---

                # 将缩放后的 PyTorch 张量转换为 PIL 图像
                pil_img = to_pil_image(resized_tensor)
                
                # 将 PIL 图像保存到内存中的字节流
                byte_stream = io.BytesIO()
                pil_img.save(byte_stream, format="jpeg", quality=args.quality)
                
                # 获取字节流的长度并收集
                byte_stream_lengths.append(len(byte_stream.getvalue()))
            except Exception as e:
                print(f"\n[警告] 处理单个图像时出错，已跳过: {e}")
                continue
    
    print("\n--- 分析完成 ---")

    # 4. 计算并打印统计结果
    if not byte_stream_lengths:
        print("未能收集到任何码流长度数据，请检查数据集和处理流程。")
        return

    lengths = np.array(byte_stream_lengths)
    count_total = len(lengths)
    count_below_20k = np.sum(lengths < 20000)
    
    print(f"{'='*40}")
    print("JPEG 码流长度统计报告")
    print(f"{'-'*40}")
    print(f"  - 总计处理图像: {count_total}")
    print(f"  - 平均长度: {np.mean(lengths):.2f} 字节")
    print(f"  - 最大长度: {np.max(lengths)} 字节")
    print(f"  - 最小长度: {np.min(lengths)} 字节")
    print(f"  - 长度中位数: {np.median(lengths):.2f} 字节")
    print(f"  - 长度小于 20000 字节的图像:")
    print(f"    - 数量: {count_below_20k}")
    print(f"    - 比例: {count_below_20k / count_total:.2%}")
    print(f"{'='*40}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分析 COCO 数据集图像的 JPEG 码流长度')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./PureT/data/coco_karpathy/validation_ids.json',
        help='COCO 数据集图像 ID 的 JSON 文件路径。'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=5000,
        help='要处理的最大样本数。设置为 0 或负数则处理全部样本。'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='处理时使用的批次大小。'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='数据加载器使用的工作进程数。'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG 编码质量 (1-100)。'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='(可选) 指向 config.yml 的路径，用于加载项目配置（如果需要）。'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 如果提供了配置文件，则加载它（尽管此脚本不直接使用大部分配置）
    if args.config and os.path.exists(args.config):
        cfg_from_file(args.config)
        print(f"已加载配置文件: {args.config}")

    analyze_lengths(args)