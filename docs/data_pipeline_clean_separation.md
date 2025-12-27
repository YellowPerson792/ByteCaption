# 数据处理管线重构 - 职责明确版

## 设计原则

**核心思想：职责分离**

1. **CocoDataset**：只负责返回JPEG字节流（224x224, quality=60）
2. **各模型的DataLoader**：自己决定如何处理字节流
   - **训练时**：直接解码为PIL图像（不损坏）
   - **评估时**：先损坏，再解码为PIL图像

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        CocoDataset                               │
│  职责：从HF加载图像 -> 标准化为224x224 -> 返回JPEG字节流        │
└────────────────────────┬────────────────────────────────────────┘
                         │ JPEG字节流 (bytes)
                         │
         ┌───────────────┼───────────────┬─────────────────┐
         │               │               │                 │
         ▼               ▼               ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│ ByteCaption    │ │ BLIP/GIT/    │ │ HF Caption  │ │ OpenRouter   │
│ DataLoader     │ │ Qwen/GLM     │ │ DataLoader  │ │ DataLoader   │
│                │ │ DataLoader   │ │             │ │              │
├────────────────┤ ├──────────────┤ ├─────────────┤ ├──────────────┤
│训练:           │ │训练:         │ │训练:        │ │评估:         │
│ 损坏 -> int32  │ │ 解码->PIL    │ │ 解码->PIL   │ │ 损坏 -> bytes│
│                │ │              │ │             │ │              │
│评估:           │ │评估:         │ │评估:        │ │              │
│ 损坏 -> int32  │ │ 损坏->解码   │ │ 损坏->解码  │ │              │
└────────────────┘ └──────────────┘ └─────────────┘ └──────────────┘
```

## 文件结构

```
PureT/datasets_/
├── coco_dataset_hf.py                # CocoDataset - 只返回JPEG字节流
├── data_loader_byteformer_coco.py    # ByteCaption + BLIP/GIT等视觉模型
├── data_loader_hf_models.py          # HF Caption模型专用（新增）
└── data_loader_hf_caption.py         # HF Caption Collator（保持不变）
```

## 各模块职责

### 1. CocoDataset (`coco_dataset_hf.py`)

**唯一职责**：返回标准化的JPEG字节流

```python
class CocoDataset:
    def __init__(self, ..., jpeg_quality: int = 60):
        self.jpeg_quality = jpeg_quality
    
    def __getitem__(self, index):
        img = self._extract_image(sample)  # PIL.Image
        jpeg_bytes = pil_to_jpeg_bytes(img, quality=self.jpeg_quality)
        return indices, gv_feat, jpeg_bytes
```

**特点**：
- ✅ 职责单一：只做图像标准化
- ✅ 统一接口：所有模型都接收相同格式
- ✅ 无需关心下游如何使用数据

### 2. ByteCaption DataLoader (`data_loader_byteformer_coco.py`)

**职责**：处理字节流用于ByteCaption模型

```python
def byteformer_collate(batch):
    """训练阶段"""
    jpeg_bytes_list = ...
    # 应用损坏（随机选一种）
    corrupted_bytes = apply_corruption(jpeg_bytes)
    # 转为int32 tensor
    int32_tensor = bytes_to_int32(corrupted_bytes)
    return ..., int32_tensor, ...

def byteformer_collate_val(batch):
    """评估阶段"""
    jpeg_bytes_list = ...
    # 应用所有损坏类型
    all_corrupted = apply_all_corruptions(jpeg_bytes)
    # 转为int32 tensor
    int32_tensors = [bytes_to_int32(b) for b in all_corrupted]
    return ..., int32_tensors, ...
```

### 3. BLIP/GIT/Qwen/GLM DataLoader (`data_loader_byteformer_coco.py`)

**职责**：处理字节流用于视觉模型

```python
def blip_collate_val(batch):
    """评估阶段"""
    jpeg_bytes_list = ...
    pil_images = []
    for jpeg_bytes in jpeg_bytes_list:
        # 应用损坏
        corrupted_bytes = apply_corruption(jpeg_bytes)
        # 解码为PIL图像
        img = Image.open(io.BytesIO(corrupted_bytes))
        pil_images.append(img)
    return ..., pil_images, ...
```

**注意**：训练时可以直接解码，不损坏：
```python
def blip_collate_train(batch):  # 如果需要
    """训练阶段"""
    jpeg_bytes_list = ...
    # 直接解码，不损坏
    pil_images = [Image.open(io.BytesIO(b)) for b in jpeg_bytes_list]
    return ..., pil_images, ...
```

### 4. HF Caption DataLoader (`data_loader_hf_models.py`) - 新增

**职责**：专门处理HF caption模型

```python
def hf_train_collate(batch):
    """训练阶段：直接解码"""
    jpeg_bytes_list = ...
    pil_images = [Image.open(io.BytesIO(b)) for b in jpeg_bytes_list]
    return indices, captions, gv_feat, pil_images

def hf_val_collate(batch):
    """评估阶段：损坏后解码"""
    jpeg_bytes_list = ...
    corrupted_and_decoded = []
    for jpeg_bytes in jpeg_bytes_list:
        corrupted_bytes = apply_corruption(jpeg_bytes)
        img = Image.open(io.BytesIO(corrupted_bytes))
        corrupted_and_decoded.append(img)
    return indices, gv_feat, corrupted_and_decoded, None
```

### 5. OpenRouter DataLoader (`data_loader_byteformer_coco.py`)

**职责**：保持字节流格式用于API调用

```python
def openrouter_collate_val(batch):
    """评估阶段：损坏但保持字节流"""
    jpeg_bytes_list = ...
    corrupted_bytes_list = []
    for jpeg_bytes in jpeg_bytes_list:
        corrupted_bytes = apply_corruption(jpeg_bytes)
        corrupted_bytes_list.append(corrupted_bytes)  # 不解码
    return ..., corrupted_bytes_list, ...
```

## 使用示例

### 示例1：训练ByteCaption

```python
from lib.config import cfg
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.datasets_.data_loader_byteformer_coco import load_train

# 创建数据集（返回字节流）
coco_set = CocoDataset(
    image_ids_path="...",
    input_seq="...",
    target_seq="...",
    gv_feat_path="",
    seq_per_img=5,
    max_feat_num=100,
    jpeg_quality=60,  # 默认60
)

# 创建dataloader（collate自动处理字节流）
train_loader = load_train(distributed=False, epoch=0, coco_set=coco_set)

for batch in train_loader:
    indices, input_seq, target_seq, gv_feat, att_feats, att_mask = batch
    # att_feats 已经是 int32 tensor
```

### 示例2：评估BLIP（有损坏）

```python
from lib.config import cfg
from PureT.datasets_.data_loader_byteformer_coco import load_val

# 配置
cfg.MODEL.TYPE = "BLIP"
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbbf"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S2"

# 创建dataloader
val_loader = load_val(
    image_ids_path="...",
    max_samples=500,
)

for batch in val_loader:
    indices, gv_feat, images, att_mask = batch
    # images 是损坏后解码的PIL图像列表
```

### 示例3：训练HF Caption模型

```python
from lib.config import cfg
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.datasets_.data_loader_hf_models import load_hf_train

# 创建数据集
coco_set = CocoDataset(
    image_ids_path="...",
    input_seq=None,
    target_seq=None,
    gv_feat_path="",
    seq_per_img=5,
    max_feat_num=100,
    return_captions=True,  # 返回captions
    jpeg_quality=60,
)

# 创建dataloader
train_loader = load_hf_train(distributed=False, epoch=0, coco_set=coco_set)

for batch in train_loader:
    indices, captions, gv_feat, images = batch
    # images 是解码后的PIL图像（训练时不损坏）
```

### 示例4：评估HF Caption模型

```python
from lib.config import cfg
from PureT.datasets_.data_loader_hf_models import load_hf_val

# 配置损坏
cfg.CORRUPTION.BYTE_STREAM_TYPES = ["rbsl"]
cfg.CORRUPTION.BYTE_STREAM_LEVEL = "S1"

# 创建dataloader
val_loader = load_hf_val(
    image_ids_path="...",
    max_samples=500,
)

for batch in val_loader:
    indices, gv_feat, images, _ = batch
    # images 是损坏后解码的PIL图像
```

## 数据流程对比

### ByteCaption 流程
```
图像 → CocoDataset → JPEG字节流
                         ↓
        训练: 随机损坏 → int32 tensor → 模型
        评估: 全部损坏 → int32 tensor → 模型
```

### 视觉模型 (BLIP/GIT/Qwen/GLM) 流程
```
图像 → CocoDataset → JPEG字节流
                         ↓
        训练: 直接解码 → PIL图像 → 模型
        评估: 损坏 → 解码 → PIL图像 → 模型
```

### HF Caption 模型流程
```
图像 → CocoDataset → JPEG字节流
                         ↓
        训练: 直接解码 → PIL图像 → HF Processor → 模型
        评估: 损坏 → 解码 → PIL图像 → HF Processor → 模型
```

### OpenRouter API 流程
```
图像 → CocoDataset → JPEG字节流
                         ↓
        评估: 损坏 → 保持字节流 → API上传
```

## 优势总结

1. **职责明确**
   - CocoDataset：只管标准化
   - DataLoader：决定如何使用字节流

2. **避免重复编码**
   - 图像只编码一次JPEG
   - 所有模型接收统一格式

3. **灵活性**
   - 训练/评估可以有不同策略
   - 新模型只需添加新collate函数

4. **代码清晰**
   - 每个模型有专门的处理逻辑
   - 易于理解和维护

5. **易于扩展**
   - 添加新损坏类型：修改corruption模块
   - 添加新模型：添加新collate函数
   - 无需修改CocoDataset

## 迁移指南

### 从旧版本迁移

1. **CocoDataset调用**：移除 `return_pil` 和 `return_jpeg_bytes` 参数
```python
# 旧版本
coco_set = CocoDataset(..., return_jpeg_bytes=True)

# 新版本（自动返回字节流）
coco_set = CocoDataset(..., jpeg_quality=60)
```

2. **DataLoader选择**：根据模型类型选择合适的loader
```python
# ByteCaption
from PureT.datasets_.data_loader_byteformer_coco import load_train, load_val

# HF Caption模型
from PureT.datasets_.data_loader_hf_models import load_hf_train, load_hf_val
```

3. **Collate函数**：已自动选择，无需修改

## 测试清单

- [ ] ByteCaption训练：字节流正确转为int32
- [ ] ByteCaption评估：多版本损坏正确生成
- [ ] BLIP评估：损坏后图像正确解码
- [ ] HF训练：图像正确解码，无损坏
- [ ] HF评估：损坏正确应用
- [ ] OpenRouter：字节流格式正确保持
- [ ] 性能测试：对比旧版本

## 注意事项

1. **所有模型统一接收字节流**：CocoDataset不再有多种返回模式
2. **训练vs评估策略不同**：训练直接解码，评估先损坏再解码
3. **损坏配置统一**：通过 `cfg.CORRUPTION` 配置
4. **内存优化**：字节流比tensor占用更少内存
