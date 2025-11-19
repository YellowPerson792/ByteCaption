#!/usr/bin/env python3
"""
独立脚本：生成Flickr8k词汇表文件

这个脚本从Flickr8k数据集的前200条数据中提取词汇，
构建词汇表文件供训练和评估使用。

使用方法:
    python generate_flickr8k_vocab.py --split train --output data/flickr8k/flickr8k_vocabulary.txt
"""

import os
import sys
import argparse
from collections import Counter
import re
import datasets

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
        
def load_flickr8k(split='train'):
    return datasets.load_dataset('jxie/flickr8k', split=split)

def basic_tokenize(text: str):
    """基础分词：转换为小写并按空格/标点分割"""
    if not isinstance(text, str):
        text = str(text)
    # 保留字母数字和撇号作为词字符
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return tokens


def build_vocabulary(split='train', max_samples=None):
    """
    从Flickr8k数据集构建词汇表（包含所有唯一词汇，不限制大小）
    
    Args:
        split: 数据集分割 ('train', 'validation', 'test')
        max_samples: 最大处理样本数，None表示处理所有样本
    
    Returns:
        list: 按频率排序的词汇列表（包含所有唯一词汇）
    """
    print(f"Loading Flickr8k {split} split...")
    try:
        ds = load_flickr8k(split)
        print(f"Loaded {len(ds)} samples from Flickr8k {split}")
    except Exception as e:
        print(f"Error loading Flickr8k dataset: {e}")
        return []
    
    # 限制样本数量
    dataset_length = min(len(ds), max_samples) if max_samples else len(ds)
    print(f"Processing {dataset_length} samples...")
    
    # 统计词频
    counter = Counter()
    processed_samples = 0
    
    for i in range(dataset_length):
        try:
            sample = ds[i]
            caps = []
            
            # 收集所有caption字段 (caption_0 到 caption_4)
            for j in range(5):
                cap_key = f'caption_{j}'
                if cap_key in sample and sample[cap_key]:
                    caps.append(sample[cap_key])
            
            # 回退：尝试其他可能的caption字段名
            if not caps:
                for alt_key in ['captions', 'caption', 'text']:
                    if alt_key in sample:
                        alt_caps = sample[alt_key]
                        if isinstance(alt_caps, str):
                            caps = [alt_caps]
                        elif isinstance(alt_caps, list):
                            caps = alt_caps
                        break
            
            # 处理captions
            if caps:
                for cap in caps:
                    if isinstance(cap, str) and cap.strip():
                        tokens = basic_tokenize(cap)
                        counter.update(tokens)
                processed_samples += 1
                
                if processed_samples % 50 == 0:
                    print(f"Processed {processed_samples}/{dataset_length} samples...")
        
        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            continue
    
    print(f"Found {len(counter)} unique tokens from {processed_samples} samples")
    
    # 选择所有词汇（按频率排序）
    most_common = [word for word, freq in counter.most_common()]
    print(f"Built vocabulary with {len(most_common)} unique words (no size limit)")
    
    # 显示统计信息
    if len(most_common) > 0:
        top_10 = counter.most_common(10)
        print(f"Top 10 most frequent words: {top_10}")
        
        # 显示词频分布统计
        total_tokens = sum(counter.values())
        print(f"Total word tokens processed: {total_tokens}")
        print(f"Average word frequency: {total_tokens / len(counter):.2f}")
        
        # 显示频率分布
        frequencies = [freq for word, freq in counter.most_common()]
        print(f"Highest frequency: {max(frequencies)}")
        print(f"Lowest frequency: {min(frequencies)}")
        print(f"Words appearing only once: {sum(1 for f in frequencies if f == 1)}")
    
    return most_common


def save_vocabulary(vocab_list, output_path):
    """
    保存词汇表到文件
    
    Args:
        vocab_list: 词汇列表
        output_path: 输出文件路径
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入词汇表文件
    # 注意：索引0保留给EOS符号'.'，实际词汇从索引1开始
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in vocab_list:
            f.write(f"{word}\n")
    
    print(f"Vocabulary saved to: {output_path}")
    print(f"Vocabulary size: {len(vocab_list)} words")
    print(f"Note: Index 0 is reserved for EOS token '.'")


def main():
    parser = argparse.ArgumentParser(description='Generate Flickr8k vocabulary file')
    parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'],
                        help='Dataset split to use (default: train)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: None for all samples)')
    parser.add_argument('--output', default='data/flickr8k/flickr8k_vocabulary.txt',
                        help='Output vocabulary file path (default: data/flickr8k/flickr8k_vocabulary.txt)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Flickr8k Vocabulary Generator")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Vocabulary size: UNLIMITED (all unique words)")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Output path: {args.output}")
    print("=" * 60)
    
    # 构建词汇表
    vocab_list = build_vocabulary(
        split=args.split,
        max_samples=args.max_samples
    )
    
    if not vocab_list:
        print("Error: Failed to build vocabulary")
        sys.exit(1)
    
    # 保存词汇表
    save_vocabulary(vocab_list, args.output)
    
    print("=" * 60)
    print("Vocabulary generation completed successfully!")
    print("=" * 60)
    
    # 验证生成的文件
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Verification: File contains {len(lines)} lines")
        if len(lines) > 0:
            print(f"First few words: {[line.strip() for line in lines[:5]]}")
        if len(lines) > 5:
            print(f"Last few words: {[line.strip() for line in lines[-3:]]}")


if __name__ == '__main__':
    main()
