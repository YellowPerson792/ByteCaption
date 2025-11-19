# ByteFormer HuggingFace Migration Project

本项目包含了将 CoreNet ByteFormer 模型迁移到 HuggingFace 框架的完整解决方案。

## 📁 项目结构

```
byteformer-hf-migration/
├── README.md                          # 项目说明文档
├── MIGRATION_SUMMARY.md              # 迁移详细总结
├── requirements.txt                   # Python依赖包
├── configs/                          # 配置文件
│   └── conv_kernel_size=4,window_sizes=[128].yaml
├── weights/                          # 预训练权重（需要手动下载）
│   └── README.md                     # 权重文件说明
├── setup.py                         # 环境设置和检查脚本
├── deploy.sh                        # 部署和GitHub设置脚本
├── utils/                           # 工具函数
│   ├── hf_adapter_utils.py          # HF适配器（来自CoreNet）
│   └── path_config.py               # 路径配置和依赖管理
├── scripts/                         # 主要脚本
│   ├── test_hf_byteformer_migration.py    # 完整迁移脚本
│   ├── hf_byteformer_usage_examples.py    # 使用示例
│   └── test_hf_byteformer.py              # 基础测试
└── examples/                        # 示例代码
    └── simple_inference.py         # 简单推理示例
```

## 🚀 快速开始

### 1. 环境检查

首先运行环境设置脚本检查所有依赖：

```bash
python setup.py
```

### 2. 安装依赖（如果需要）

```bash
pip install -r requirements.txt
```

### 3. 配置CoreNet路径

如果CoreNet不在默认位置，设置环境变量：

```bash
export CORENET_PATH=/path/to/your/corenet
```

### 4. 准备权重文件

确保权重文件在正确位置：
- `weights/imagenet_jpeg_q60_k4_w128.pt`

### 5. 运行迁移和测试

```bash
# 完整迁移测试
python scripts/test_hf_byteformer_migration.py

# 简单推理示例
python examples/simple_inference.py

# 详细使用示例
python scripts/hf_byteformer_usage_examples.py
```

## 📚 主要功能

- ✅ **完整的框架迁移**：从 CoreNet 到 HuggingFace
- ✅ **预训练权重加载**：直接加载原始权重文件
- ✅ **HF生态兼容**：支持所有 HuggingFace 工具和API
- ✅ **推理验证**：完整的推理测试和验证
- ✅ **训练支持**：可用于 HuggingFace Trainer

## 🔧 技术细节

### 模型规格
- **架构**: ByteFormer Tiny
- **嵌入维度**: 192
- **变换器层数**: 12
- **注意力头数**: 3
- **词汇表大小**: 257
- **卷积核大小**: 4
- **窗口大小**: 128

### 依赖要求
- PyTorch >= 2.0
- Transformers >= 4.21
- CoreNet 框架（用于导入工具）

## 📖 使用说明

详细的使用方法和API说明请参考：
- [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) - 完整迁移文档
- [scripts/hf_byteformer_usage_examples.py](scripts/hf_byteformer_usage_examples.py) - 详细使用示例

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📄 许可证

本项目遵循原 CoreNet 项目的许可证条款。

## 🙏 致谢

- Apple CoreNet 团队提供的原始 ByteFormer 实现
- HuggingFace 团队提供的优秀框架
