#!/usr/bin/env python3
"""
简单的模型下载脚本
使用 modelscope 下载模型
"""

from modelscope import snapshot_download
import os

def download_internvl():
    """下载模型"""
    
    model_id = "Salesforce/blip-image-captioning-base"
    local_dir = "blip-image-captioning-base"
    
    print(f"开始从 ModelScope 下载模型: {model_id}")
    print(f"保存路径: {local_dir}")
    
    try:
        snapshot_download(
            model_id=model_id,
            cache_dir=local_dir,
            revision='master',
        )
        print(f"\n✓ 模型下载完成！")
        print(f"  路径: {os.path.abspath(local_dir)}")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_internvl()
