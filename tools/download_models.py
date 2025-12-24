#!/usr/bin/env python3
"""
简单的模型下载脚本
使用 hf-mirror 镜像加速下载
"""

from huggingface_hub import snapshot_download
import os

def download_internvl():
    """下载模型"""
    
    # 设置 hf-mirror 镜像加速
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    model_id = "zai-org/GLM-4.6V-Flash"
    local_dir = "./GLM-4.6V-Flash"
    
    print(f"使用镜像: {os.environ.get('HF_ENDPOINT', 'huggingface.co')}")
    print(f"开始下载模型: {model_id}")
    print(f"保存路径: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )
        print(f"\n✓ 模型下载完成！")
        print(f"  路径: {os.path.abspath(local_dir)}")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_internvl()
