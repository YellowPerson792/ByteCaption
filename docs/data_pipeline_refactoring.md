# 数据处理管线重构说明

## 重构目标

将数据处理管线重新组织，使得：
1. **CocoDataset** 统一处理图像为尺寸224×224、质量60的JPEG，返回字节流
2. 各种类型的 **dataloader collate函数** 根据模型类型处理字节流：
   - 如果是ByteCaption字节流模型，就直接应用损坏并输入
   - 如果是BLIP/GIT/QWEN等其他视觉模型，就先损坏，然后重新解码为PIL图像后输入
   - 如果是HF模型且无损坏（清洁图像），直接解码为PIL图像

## 修改的文件

### 1. `PureT/datasets_/coco_dataset_hf.py`

**新增函数：**
```python
def pil_to_jpeg_bytes(img: Image.Image, quality: int = 60) -> bytes:
    """将PIL图像转换为JPEG字节流（224x224）"""
```

**修改 CocoDataset.__init__：**
- 新增参数 `return_jpeg_bytes: bool = False` - 返回JPEG字节流模式
- 新增参数 `jpeg_quality: int = 60` - JPEG压缩质量

**修改 CocoDataset.__getitem__：**
- 根据 `return_jpeg_bytes` 标志选择返回格式：
  - `return_jpeg_bytes=True`: 返回JPEG字节流（bytes）
  - `return_pil=True`: 返回PIL图像（用于HF清洁模式）
  - 默认: 返回Tensor（用于传统视觉模型，向后兼容）

### 2. `PureT/datasets_/data_loader_byteformer_coco_v2.py` (新文件)

创建了全新的dataloader模块，包含以下collate函数：

#### `byteformer_collate(batch)` - 训练阶段
- **输入**: JPEG字节流
- **处理**: 应用随机损坏（训练时随机选一种）-> 转为int32 tensor
- **输出**: ByteFormer可用的padded int32 tensor

#### `byteformer_collate_val(batch)` - ByteCaption验证
- **输入**: JPEG字节流
- **处理**: 应用所有损坏类型 -> 转为int32 tensor
- **输出**: ByteFormer可用的padded int32 tensor（多版本）
- **特点**: 支持样本增强（1->N），自动复制元数据

#### `blip_collate_val(batch)` - BLIP/GIT/视觉模型验证
- **输入**: JPEG字节流
- **处理**: 应用损坏 -> 解码为PIL图像
- **输出**: PIL图像列表
- **特点**: 
  - 自动检查宽高比，调整异常图像
  - 保存前5个损坏样本用于调试
  - 解码失败返回None占位

#### `hf_collate_val(batch)` - HF清洁模式验证
- **输入**: PIL图像（CocoDataset直接返回）
- **处理**: 无
- **输出**: PIL图像列表
- **用途**: 用于无损坏的HF模型评估

#### `openrouter_collate_val(batch)` - OpenRouter API验证
- **输入**: JPEG字节流
- **处理**: 应用损坏 -> 保持字节流格式
- **输出**: JPEG字节流列表
- **用途**: 用于API调用，直接上传损坏的JPEG

### 3. `load_val()` 函数逻辑

自动判断使用哪种collate模式：

```python
# 判断逻辑
is_openrouter = "openrouter" in model_type or "gpt" in model_type
is_hf = "blip" in model_type or "git" in model_type or "qwen" in model_type ...
use_clean_hf = is_hf and level in {"S0", "M0"}

# CocoDataset配置
coco_set = CocoDataset(
    ...,
    return_pil=use_clean_hf,           # HF清洁模式返回PIL
    return_jpeg_bytes=(not use_clean_hf),  # 其他模式返回字节流
    jpeg_quality=60,
)

# 选择collate函数
if is_openrouter:
    collate_fn = openrouter_collate_val
elif use_clean_hf:
    collate_fn = hf_collate_val
elif is_hf:
    collate_fn = blip_collate_val
else:
    collate_fn = byteformer_collate_val
```

## 数据流程图

### ByteCaption 模型流程
```
原始图像 (HF Dataset)
    ↓
CocoDataset (return_jpeg_bytes=True)
    ↓
JPEG字节流 (224x224, quality=60)
    ↓
byteformer_collate / byteformer_collate_val
    ↓
应用损坏 (RBBF/RBSL/Metadata)
    ↓
转换为 int32 tensor
    ↓
ByteFormer collate padding
    ↓
模型输入
```

### BLIP/GIT 等视觉模型流程
```
原始图像 (HF Dataset)
    ↓
CocoDataset (return_jpeg_bytes=True)
    ↓
JPEG字节流 (224x224, quality=60)
    ↓
blip_collate_val
    ↓
应用损坏 (RBBF/RBSL/Metadata)
    ↓
解码为 PIL 图像
    ↓
HF Processor处理
    ↓
模型输入
```

### HF 清洁模式流程
```
原始图像 (HF Dataset)
    ↓
CocoDataset (return_pil=True)
    ↓
PIL 图像 (224x224)
    ↓
hf_collate_val
    ↓
HF Processor处理
    ↓
模型输入
```

### OpenRouter API 流程
```
原始图像 (HF Dataset)
    ↓
CocoDataset (return_jpeg_bytes=True)
    ↓
JPEG字节流 (224x224, quality=60)
    ↓
openrouter_collate_val
    ↓
应用损坏 (RBBF/RBSL/Metadata)
    ↓
保持字节流格式
    ↓
API上传
```

## 优势

1. **职责分离**: CocoDataset只负责图像标准化，collate函数负责损坏处理
2. **避免重复编码**: 图像只在CocoDataset编码一次JPEG
3. **灵活的损坏处理**: 不同模型可以有不同的损坏策略
4. **代码复用**: 损坏逻辑集中在`ByteStreamCorrupter`中
5. **向后兼容**: 保留了原有的tensor返回模式

## 使用示例

### 训练ByteCaption模型
```python
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_train

coco_set = CocoDataset(
    image_ids_path="path/to/train_ids.json",
    input_seq="path/to/train_input.pkl",
    target_seq="path/to/train_target.pkl",
    gv_feat_path="",
    seq_per_img=5,
    max_feat_num=100,
    return_jpeg_bytes=True,  # 返回字节流
    jpeg_quality=60,
)

train_loader = load_train(distributed=False, epoch=0, coco_set=coco_set)
```

### 评估ByteCaption模型
```python
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_val

# 配置损坏参数
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbbf", "rbsl"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S2"

val_loader = load_val(
    image_ids_path="path/to/val_ids.json",
    max_samples=500,
)
```

### 评估BLIP模型
```python
# 设置模型类型
cfg.MODEL.TYPE = "BLIP"
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbbf"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S1"

val_loader = load_val(
    image_ids_path="path/to/val_ids.json",
    max_samples=500,
)
# 自动使用 blip_collate_val
```

### 评估HF模型（清洁图像）
```python
cfg.MODEL.TYPE = "HF_QWEN"
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S0"  # 无损坏

val_loader = load_val(
    image_ids_path="path/to/val_ids.json",
    max_samples=500,
)
# 自动使用 hf_collate_val，CocoDataset返回PIL图像
```

## 迁移指南

### 从旧版本迁移

1. **更新import**:
```python
# 旧版本
from PureT.datasets_.data_loader_byteformer_coco import load_train, load_val

# 新版本
from PureT.datasets_.data_loader_byteformer_coco_v2 import load_train, load_val
```

2. **更新CocoDataset调用**:
```python
# ByteCaption模型 - 使用字节流
coco_set = CocoDataset(
    ...,
    return_jpeg_bytes=True,  # 新增
    jpeg_quality=60,         # 新增
)

# HF清洁模式 - 使用PIL
coco_set = CocoDataset(
    ...,
    return_pil=True,         # 已有，用于HF
)
```

3. **损坏配置保持不变**:
```python
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbbf", "rbsl"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S2"
```

## 注意事项

1. **字节流长度统计**: 全局变量 `_BYTE_STREAM_LENGTHS` 收集所有字节流长度，用于分析
2. **损坏样本保存**: `blip_collate_val` 会自动保存前5个损坏样本到 `./evaluation_samples/`
3. **宽高比限制**: 通过环境变量 `BC_MAX_ASPECT_RATIO` 控制（默认150）
4. **并发安全**: 使用 `worker_init_fn` 确保多进程DataLoader的随机种子独立

## 待办事项

- [ ] 测试新版本dataloader与现有训练/评估脚本的兼容性
- [ ] 性能基准测试（对比旧版本）
- [ ] 完善错误处理和日志
- [ ] 添加单元测试
- [ ] 更新文档和示例代码
