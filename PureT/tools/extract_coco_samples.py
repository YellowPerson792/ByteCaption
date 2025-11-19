#!/usr/bin/env python3
"""
COCO数据提取脚本 - 专门处理HuggingFace格式数据
提取前500张验证集图像和对应的captions
"""

import os
import json
from datasets import load_from_disk
from PIL import Image

def main():
    print("=" * 60)
    print("COCO Validation Dataset Sample Extractor (HuggingFace)")
    print("=" * 60)
    
    # 数据路径
    data_path = "/root/autodl-tmp/ByteCaption/PureT/data/coco_karpathy/AbdoTW___coco_2014_karpathy/validation"
    output_dir = "/root/autodl-tmp/ByteCaption/PureT/sample_data_500"
    images_dir = os.path.join(output_dir, "images")
    
    # 创建输出目录
    os.makedirs(images_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # 加载数据集
        print("Loading dataset...")
        dataset = load_from_disk(data_path)
        
        # 获取验证集
        if isinstance(dataset, dict):
            if 'validation' in dataset:
                val_dataset = dataset['validation']
            elif 'test' in dataset:
                val_dataset = dataset['test']
            else:
                # 取第一个可用的split
                val_dataset = list(dataset.values())[0]
        else:
            val_dataset = dataset
        
        print(f"Dataset loaded: {len(val_dataset)} samples")
        
        # 检查数据结构
        if len(val_dataset) > 0:
            sample = val_dataset[0]
            print("Data structure:")
            for key, value in sample.items():
                if hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"  {key}: {type(value)} (length: {len(value)})")
                else:
                    print(f"  {key}: {type(value)}")
        
        # 提取前500个样本
        num_samples = min(500, len(val_dataset))
        sample_info = []
        
        for i in range(num_samples):
            try:
                sample = val_dataset[i]
                
                # 获取图像
                image = sample.get('image')
                if image is None:
                    print(f"Warning: No image found for sample {i}")
                    continue
                
                # 保存图像
                image_filename = f"coco_val_{i:05d}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                if hasattr(image, 'save'):
                    # PIL Image
                    image.save(image_path, 'JPEG')
                else:
                    print(f"Warning: Unexpected image format for sample {i}: {type(image)}")
                    continue
                
                # 获取caption(s)
                captions = []
                for caption_key in ['text', 'caption', 'captions']:
                    if caption_key in sample:
                        cap = sample[caption_key]
                        if isinstance(cap, list):
                            captions.extend(cap)
                        elif cap:
                            captions.append(cap)
                        break
                
                # 收集样本信息
                sample_info.append({
                    "id": i,
                    "image_filename": image_filename,
                    "captions": captions,
                    "original_keys": list(sample.keys())
                })
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{num_samples} images...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # 保存样本信息到JSON文件
        info_file = os.path.join(output_dir, "sample_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_samples": len(sample_info),
                "data_source": "huggingface",
                "samples": sample_info
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nSample data saved successfully!")
        print(f"Total samples: {len(sample_info)}")
        print(f"Images directory: {images_dir}")
        print(f"Info file: {info_file}")
        
        # 显示一些统计信息
        if sample_info:
            total_captions = sum(len(s['captions']) for s in sample_info)
            avg_captions = total_captions / len(sample_info) if sample_info else 0
            print(f"Total captions: {total_captions}")
            print(f"Average captions per image: {avg_captions:.2f}")
            
            # 显示第一个样本的信息
            first_sample = sample_info[0]
            print(f"\nFirst sample example:")
            print(f"  Image: {first_sample['image_filename']}")
            print(f"  Captions: {len(first_sample['captions'])}")
            for j, caption in enumerate(first_sample['captions'][:3]):  # 显示前3个caption
                print(f"    {j+1}: {caption}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()