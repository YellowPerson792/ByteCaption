# 数据处理管线重构 - 快速开始

## 概述

数据处理管线已重构，实现了以下目标：
- **CocoDataset** 统一返回标准化的JPEG字节流（224×224, quality=60）
- **各类dataloader** 根据模型类型灵活处理字节流
- **损坏处理** 集中在collate函数中，避免重复编码

## 主要改动

### 1. CocoDataset 新增返回模式

```python
from PureT.datasets_.coco_dataset_hf import CocoDataset

# 模式1: 返回JPEG字节流（用于ByteCaption和需要损坏的模型）
dataset = CocoDataset(
    ...,
    return_jpeg_bytes=True,
    jpeg_quality=60,
)
# 返回: (indices, gv_feat, jpeg_bytes: bytes)

# 模式2: 返回PIL图像（用于HF清洁模式）
dataset = CocoDataset(
    ...,
    return_pil=True,
)
# 返回: (indices, gv_feat, pil_image: PIL.Image)

# 模式3: 返回Tensor（默认，向后兼容）
dataset = CocoDataset(...)
# 返回: (indices, gv_feat, tensor: torch.Tensor)
```

### 2. 新的DataLoader模块

使用新版本dataloader：
```python
# 导入新模块
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_train, load_val

# 训练
train_loader = load_train(distributed=False, epoch=0, coco_set=dataset)

# 验证（自动选择合适的collate函数）
val_loader = load_val(
    image_ids_path="./PureT/data/coco_karpathy/val_image_ids.txt",
    max_samples=500,
)
```

## 使用示例

### 示例1: 训练ByteCaption模型

```python
from lib.config import cfg
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_train

# 配置
cfg.TRAIN.BATCH_SIZE = 32
cfg.DATA_LOADER.NUM_WORKERS = 4
cfg.DATA_LOADER.SEQ_PER_IMG = 5

# 创建数据集
coco_set = CocoDataset(
    image_ids_path="./PureT/data/coco_karpathy/train_image_ids.txt",
    input_seq="./PureT/data/coco/train_input_seq.pkl",
    target_seq="./PureT/data/coco/train_target_seq.pkl",
    gv_feat_path="",
    seq_per_img=5,
    max_feat_num=100,
    return_jpeg_bytes=True,  # 关键：返回字节流
    jpeg_quality=60,
)

# 创建dataloader
train_loader = load_train(distributed=False, epoch=0, coco_set=coco_set)

# 训练循环
for batch in train_loader:
    indices, input_seq, target_seq, gv_feat, att_feats, att_mask = batch
    # att_feats 是 padded int32 tensor，可直接输入ByteCaption模型
    ...
```

### 示例2: 评估ByteCaption模型（有损坏）

```python
from lib.config import cfg
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_val

# 配置损坏参数
cfg.MODEL.TYPE = "ByteCaption"
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbbf", "rbsl"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S2"
cfg.TEST.BATCH_SIZE = 16

# 创建dataloader（自动使用byteformer_collate_val）
val_loader = load_val(
    image_ids_path="./PureT/data/coco_karpathy/val_image_ids.txt",
    max_samples=500,
)

# 评估循环
for batch in val_loader:
    indices, gv_feat, att_feats, att_mask = batch
    # att_feats 包含多个损坏版本（如果配置了多种损坏类型）
    ...
```

### 示例3: 评估BLIP模型（有损坏）

```python
from lib.config import cfg
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_val

# 配置
cfg.MODEL.TYPE = "BLIP"  # 关键：指定BLIP模型
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbbf"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S1"

# 创建dataloader（自动使用blip_collate_val）
val_loader = load_val(
    image_ids_path="./PureT/data/coco_karpathy/val_image_ids.txt",
    max_samples=500,
)

# 评估循环
for batch in val_loader:
    indices, gv_feat, images, att_mask = batch
    # images 是 PIL 图像列表（已从损坏的字节流解码）
    ...
```

### 示例4: 评估HF模型（无损坏）

```python
from lib.config import cfg
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_val

# 配置
cfg.MODEL.TYPE = "HF_QWEN"
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S0"  # 无损坏

# 创建dataloader（自动使用hf_collate_val）
val_loader = load_val(
    image_ids_path="./PureT/data/coco_karpathy/val_image_ids.txt",
    max_samples=500,
)

# 评估循环
for batch in val_loader:
    indices, gv_feat, images, att_mask = batch
    # images 是清洁的 PIL 图像列表
    ...
```

## 测试

运行测试脚本验证重构：

```bash
python tools/test_data_pipeline_refactor.py
```

测试内容包括：
1. PIL到JPEG字节流的转换
2. CocoDataset的三种返回模式
3. 各种collate函数的正确性
4. 损坏管线的功能

## 迁移指南

### 从旧版本迁移

1. **更新import语句**:
```python
# 旧版本
from PureT.datasets_.data_loader_byteformer_coco import load_train, load_val

# 新版本
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_train, load_val
```

2. **ByteCaption训练/评估**: 添加 `return_jpeg_bytes=True` 参数
```python
coco_set = CocoDataset(
    ...,
    return_jpeg_bytes=True,  # 新增
    jpeg_quality=60,         # 新增
)
```

3. **其他模型**: 配置会自动选择合适的模式，通常无需修改

## 文件结构

```
PureT/datasets_/
├── coco_dataset_hf.py              # 修改：新增字节流返回模式
├── data_loader_byteformer_coco.py  # 旧版本（保留）
└── data_loader_byteformer_coco_v2.py  # 新版本

docs/
└── data_pipeline_refactoring.md   # 详细文档

tools/
└── test_data_pipeline_refactor.py  # 测试脚本
```

## 核心优势

1. **避免重复编码**: 图像只编码一次JPEG
2. **职责分离**: Dataset负责标准化，Collate负责损坏
3. **灵活性**: 不同模型可以有不同的损坏策略
4. **代码复用**: 损坏逻辑集中管理
5. **向后兼容**: 保留了原有的tensor模式

## 性能考虑

- JPEG编码在Dataset中只进行一次
- 损坏处理在collate函数中批量进行
- 支持多进程DataLoader和数据预取
- 字节流比图像tensor占用更少内存

## 常见问题

**Q: 为什么要统一使用JPEG字节流？**
A: 这样可以避免在每次损坏时重新编码图像，同时保证所有模型使用相同质量的JPEG（公平比较）。

**Q: 如果我想使用不同的JPEG质量怎么办？**
A: 在创建CocoDataset时设置 `jpeg_quality` 参数。

**Q: 损坏的图像解码失败会怎样？**
A: blip_collate_val 会返回 None 占位符，评估器需要能够处理这种情况。

**Q: 如何查看损坏后的图像？**
A: blip_collate_val 会自动保存前5个损坏样本到 `./evaluation_samples/` 目录。

## 更多信息

查看详细文档：[docs/data_pipeline_refactoring.md](./docs/data_pipeline_refactoring.md)
