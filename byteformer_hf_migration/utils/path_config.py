#!/usr/bin/env python3
"""
路径配置和依赖管理
处理CoreNet框架的路径依赖问题
"""

import sys
import os
from pathlib import Path

def setup_corenet_path():
    """
    设置CoreNet路径，支持多种安装方式
    """
    # 可能的CoreNet路径
    possible_paths = [
        "D:\\MLLMs\\corenet",
        "/root/autodl-tmp/corenet",  # 开发环境
        "/opt/corenet",              # 系统安装
        os.path.expanduser("~/corenet"),  # 用户目录
        os.path.join(os.getcwd(), "corenet"),  # 当前目录
    ]
    
    # 检查环境变量
    corenet_path = os.environ.get("CORENET_PATH")
    if corenet_path:
        possible_paths.insert(0, corenet_path)
    
    # 寻找可用的CoreNet路径
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "corenet")):
            if path not in sys.path:
                sys.path.insert(0, path)
            print(f"✓ CoreNet路径已设置: {path}")
            return path
    
    # 如果没有找到，提供安装指导
    print("❌ 未找到CoreNet安装路径")
    print("\n请选择以下方式之一来解决:")
    print("1. 设置环境变量: export CORENET_PATH=/path/to/corenet")
    print("2. 将CoreNet克隆到当前目录: git clone https://github.com/apple/corenet.git")
    print("3. 将CoreNet路径添加到Python路径")
    
    raise ImportError("CoreNet框架未找到，请参考上述说明进行配置")

def get_config_file_path():
    """
    获取配置文件的完整路径
    """
    current_dir = Path(__file__).parent
    config_file = current_dir.parent / "configs" / "conv_kernel_size=4,window_sizes=[128].yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_file}")
    
    return str(config_file.absolute())

def get_weights_file_path():
    """
    获取权重文件的完整路径
    """
    current_dir = Path(__file__).parent
    weights_file = current_dir.parent / "weights" / "imagenet_jpeg_q60_k4_w128.pt"
    
    if not weights_file.exists():
        print(f"❌ 权重文件未找到: {weights_file}")
        print("\n请将权重文件复制到 weights/ 目录:")
        print(f"  cp /root/autodl-tmp/corenet/weights/imagenet_jpeg_q60_k4_w128.pt {weights_file}")
        raise FileNotFoundError(f"权重文件未找到: {weights_file}")
    
    return str(weights_file.absolute())

def check_dependencies():
    """
    检查必要的依赖包
    """
    required_packages = [
        "torch",
        "transformers",
        "numpy",
        "yaml"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        raise ImportError(f"缺少依赖包: {missing_packages}")
    
    print("✓ 所有依赖包已安装")

if __name__ == "__main__":
    print("=== 路径配置检查 ===")
    try:
        check_dependencies()
        setup_corenet_path()
        config_path = get_config_file_path()
        weights_path = get_weights_file_path()
        
        print(f"✓ 配置文件: {config_path}")
        print(f"✓ 权重文件: {weights_path}")
        print("✓ 所有路径配置正确")
        
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        sys.exit(1)
