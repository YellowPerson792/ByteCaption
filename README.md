# ByteCaption: 基于字节流的图像描述生成

ByteCaption 是一个创新的图像描述生成项目，结合了 Apple CoreNet 的 ByteFormer 和 PureT 模型，实现从 JPEG 压缩码流直接生成图像描述的端到端训练与评估。

## 📖 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目架构](#项目架构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [训练模型](#训练模型)
- [模型评估](#模型评估)
- [模型变体](#模型变体)
- [参考文献](#参考文献)
- [致谢](#致谢)

## 项目简介

ByteCaption 探索了一种新颖的图像描述生成方法：**直接从 JPEG 压缩码流生成描述文本**，而不是传统的从解码后的图像像素生成。该方法具有以下优势：

- **节省计算资源**：跳过图像解码步骤，直接处理压缩数据
- **提升鲁棒性**：天然抵抗图像压缩损失和传输损坏
- **创新架构**：结合字节级编码器（ByteFormer）和 Transformer 解码器（PureT）

本项目基于以下研究成果：
- **ByteFormer**：Apple 开源的字节级 Transformer，可直接处理文件字节流（来自论文 "Bytes Are All You Need: Transformers Operating Directly on File Bytes"）
- **PureT**：端到端 Transformer 图像描述模型（来自论文 "End-to-End Transformer Based Model for Image Captioning"，AAAI 2022）

## 核心特性

- ✅ **字节流处理**：直接从 JPEG 码流生成图像描述，无需解码
- ✅ **多种增强策略**：支持字节损坏、掩码、置换、噪声等数据增强
- ✅ **灵活的模型架构**：支持 ByteFormer、BLIP、GIT、Qwen-VL、GLM、InternVL、Ministral 等多种视觉-语言模型
- ✅ **HuggingFace 集成**：完整的 HuggingFace Transformers 生态支持
- ✅ **全面的评估指标**：支持 BLEU、METEOR、ROUGE-L、CIDEr、SPICE 等标准指标
- ✅ **分布式训练**：支持单机多卡和多机多卡训练

## 项目架构

```
ByteCaption/
├── README.md                          # 本文档
├── requirements.txt                   # Python 依赖
├── corenet/                          # Apple CoreNet 框架
│   └── data/
│       ├── transforms/
│       │   └── image_bytes.py       # 字节级数据增强
│       └── collate_fns/
│           └── byteformer_collate_functions.py  # ByteFormer 专用 collate
├── PureT/                            # PureT 模型和训练代码
│   ├── main.py                       # 训练入口（支持 COCO/Flickr8k）
│   ├── main_val.py                   # 验证入口
│   ├── byteformer_immigration.py     # ByteFormer HF 封装
│   ├── datasets_/                    # 数据集封装
│   │   ├── coco_dataset_hf.py       # COCO 数据集（HuggingFace）
│   │   └── data_loader_byteformer_coco.py  # ByteFormer collate
│   └── experiments/                  # 实验配置和输出
│       ├── ByteCaption_XE/          # 标准 ByteFormer 配置
│       ├── ByteCaption_XE_blip/     # BLIP 模型配置
│       ├── ByteCaption_XE_glm/      # GLM 模型配置
│       └── ByteCaption_XE_qwen/     # Qwen-VL 模型配置
├── byteformer_hf_migration/          # ByteFormer HuggingFace 迁移
│   ├── README.md                     # 迁移文档
│   ├── configs/                      # ByteFormer 配置文件
│   ├── weights/                      # 预训练权重（需下载）
│   ├── scripts/                      # 训练和推理脚本
│   └── utils/                        # 工具函数
└── tools/                            # 训练工具脚本
    ├── train_hf_trainer.py          # Qwen-VL 训练脚本
    ├── train_glm_hf_trainer.py      # GLM 训练脚本
    ├── train_internvl_swift.py      # InternVL Swift 训练
    └── train_ministral3_hf_trainer.py  # Ministral 训练脚本
```

## 环境配置

### 系统要求

- Python 3.9+ (Linux) 或 3.10+ (macOS)
- PyTorch >= 2.3.0
- CUDA 11.8+ (GPU 训练)
- 推荐至少 16GB 显存用于训练

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/YellowPerson792/ByteCaption.git
cd ByteCaption
```

2. **创建虚拟环境（推荐）**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
# venv\Scripts\activate  # Windows
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **设置环境变量**

```bash
export PYTHONPATH=$(pwd)
```

5. **安装 Git LFS（可选，用于下载大文件）**

```bash
# Linux
sudo apt install git-lfs
git lfs install
git lfs pull

# macOS
brew install git-lfs
git lfs install
git lfs pull
```

## 数据准备

### 1. COCO 数据集

数据集将在首次运行时自动下载到：
```
PureT/data/coco_karpathy/AbdoTW___coco_2014_karpathy/{train,validation,test}
```

### 2. 图像 ID 列表

确保以下文件存在：
- `PureT/data/coco_karpathy/train_ids.json`
- `PureT/data/coco_karpathy/validation_ids.json`

### 3. 词表生成

首次运行训练时会自动生成词表文件：
- `PureT/data/coco_karpathy/coco_vocabulary.txt`

### 4. ByteFormer 预训练权重

下载 ByteFormer 预训练权重并放置在：
```
byteformer_hf_migration/weights/imagenet_jpeg_q60_k4_w128.pt
```

## 快速开始

### 方式一：使用 ByteFormer + PureT（推荐）

这是项目的核心方法，直接从 JPEG 码流生成描述。

```bash
python PureT/main.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --eval_steps 600 \
  --val_samples 50 \
  --load_weights \
  --freeze_backbone \
  --disable_wandb
```

**参数说明：**
- `--folder`: 实验配置和输出目录
- `--dataset`: 数据集名称（coco 或 flickr8k）
- `--eval_steps`: 每隔多少步评估一次
- `--val_samples`: 验证时使用的样本数（0 表示全部）
- `--load_weights`: 加载预训练的 ByteFormer 权重
- `--freeze_backbone`: 冻结 ByteFormer 编码器，只训练解码器
- `--disable_wandb`: 禁用 Weights & Biases 日志

### 方式二：使用现代视觉-语言模型

#### Qwen-VL 训练

```bash
python tools/train_hf_trainer.py \
  --folder PureT/experiments/ByteCaption_XE_qwen \
  --dataset coco \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --processor_id Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ./Qwen3-VL-8B-Instruct \
  --train_samples 0 \
  --val_samples 10 \
  --lora_r 16 \
  --lora_alpha 32 \
  --attn_implementation flash_attention_2 \
  --disable_wandb
```

#### GLM 训练

```bash
python tools/train_glm_hf_trainer.py \
  --folder PureT/experiments/ByteCaption_XE_glm \
  --dataset coco \
  --model_id THUDM/glm-4v-9b \
  --local_dir ./glm-4v-9b \
  --lora_r 8 \
  --lora_alpha 16
```

#### InternVL 训练（使用 Swift 框架）

```bash
python tools/train_internvl_swift.py \
  --folder PureT/experiments/ByteCaption_XE_internvl \
  --dataset coco \
  --model_id OpenGVLab/InternVL2-8B \
  --num_train_epochs 3
```

## 训练模型

### 单机单卡训练

```bash
CUDA_VISIBLE_DEVICES=0 python PureT/main.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --eval_steps 600 \
  --val_samples 50 \
  --load_weights \
  --freeze_backbone
```

### 单机多卡训练（DDP）

```bash
torchrun --nproc_per_node=4 --master_port=12355 PureT/main.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --eval_steps 600 \
  --val_samples 50 \
  --load_weights \
  --freeze_backbone
```

### 端到端微调（不冻结编码器）

```bash
# 注意：移除 --freeze_backbone 参数以进行端到端微调
python PureT/main.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --eval_steps 600 \
  --val_samples 50 \
  --load_weights
```

### 训练配置

修改 `PureT/experiments/ByteCaption_XE/config_coco.yml` 来自定义训练参数：

```yaml
# 主要配置项
SOLVER:
  MAX_EPOCH: 30
  BASE_LR: 0.0001
  GRAD_CLIP: 5.0

DATA_LOADER:
  BATCH_SIZE: 16
  NUM_WORKERS: 4

MODEL:
  TYPE: PureT_byteformer  # 或 PureT (使用 BLIP)
  D_MODEL: 512
  HEAD: 8
  ENC_LAYERS: 6
  DEC_LAYERS: 6
```

## 模型评估

### 验证模型性能

```bash
python PureT/main_val.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --val_samples 500 \
  --resume -1 \
  --disable_wandb
```

**参数说明：**
- `--resume -1`: 加载 best_model.pth
- `--resume N`: 加载第 N 个 epoch 的模型

### 评估输出

评估结果将保存在：
- `PureT/experiments/ByteCaption_XE/result/`：包含 COCO 格式的预测结果
- `PureT/experiments/ByteCaption_XE/log.txt`：训练和评估日志

评估指标包括：
- **BLEU-1/2/3/4**：N-gram 精确度
- **METEOR**：考虑同义词和词干的指标
- **ROUGE-L**：最长公共子序列
- **CIDEr**：共识度指标
- **SPICE**：基于场景图的语义指标

## 模型变体

### 1. ByteFormer + PureT（核心方法）

直接从 JPEG 码流生成描述：

```
配置目录：PureT/experiments/ByteCaption_XE/
特点：字节级处理，鲁棒性强
```

### 2. BLIP 模型

传统的图像-文本模型：

```
配置目录：PureT/experiments/ByteCaption_XE_blip/
特点：成熟的预训练模型，性能稳定
```

### 3. GIT 模型

Microsoft 的生成式图像-文本 Transformer：

```
模型目录：git-base-coco/
特点：简单高效的生成式架构
```

### 4. Qwen-VL 系列

阿里云的多模态大模型：

```
配置目录：PureT/experiments/ByteCaption_XE_qwen/
特点：强大的多模态理解能力，支持 LoRA 微调
```

### 5. GLM-4V 系列

清华大学的多模态大模型：

```
配置目录：PureT/experiments/ByteCaption_XE_glm/
特点：中英文双语能力，对话式交互
```

### 6. InternVL 系列

上海 AI 实验室的视觉-语言模型：

```
配置目录：PureT/experiments/ByteCaption_XE_internvl/
特点：高分辨率图像理解，性能优秀
```

## 高级功能

### 字节流增强

项目支持多种字节流增强策略（在 `corenet/data/transforms/image_bytes.py` 中实现）：

- **PIL 编码参数**：质量、格式、优化等
- **字节损坏**：模拟传输错误
- **字节掩码**：随机遮盖部分字节
- **字节置换**：打乱字节顺序
- **字节噪声**：添加随机噪声

### 调试工具

1. **分析码流长度**

```bash
python PureT/analyze_stream_length.py \
  --dataset_path PureT/data/coco_karpathy/validation_ids.json \
  --max_samples 2000 \
  --quality 95
```

2. **编码对比**

```bash
python PureT/compare_encoding.py
# 查看 encoding_compare_samples/ 目录下的对比结果
```

3. **查看评估样本**

训练/验证时会自动保存一些样本到 `evaluation_samples/` 目录，可视化损坏后的图像重建效果。

## 参考文献

### ByteFormer

```bibtex
@article{patil2023bytes,
  title={Bytes Are All You Need: Transformers Operating Directly on File Bytes},
  author={Patil, Maxwell Horton and others},
  journal={arXiv preprint arXiv:2306.00238},
  year={2023}
}
```

### PureT

```bibtex
@inproceedings{wang2022puret,
  title={End-to-End Transformer Based Model for Image Captioning},
  author={Wang, Yiyu and Xu, Jungang and Sun, Yingfei},
  booktitle={AAAI},
  year={2022}
}
```

### CoreNet

```bibtex
@inproceedings{mehta2022cvnets,
  author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad},
  title = {CVNets: High Performance Library for Computer Vision},
  year = {2022},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  series = {MM '22}
}
```

## 致谢

本项目基于以下优秀的开源项目：

- [Apple CoreNet](https://github.com/apple/corenet)：提供 ByteFormer 实现和训练框架
- [PureT](https://github.com/yiyu-wang/PureT)：提供端到端 Transformer 图像描述模型
- [HuggingFace Transformers](https://github.com/huggingface/transformers)：提供模型生态和工具
- [COCO Caption](https://github.com/tylin/coco-caption)：提供评估工具

感谢以上项目的作者和贡献者！

## License

本项目遵循 [MIT License](LICENSE)。

---

如有问题或建议，欢迎提交 Issue 或 Pull Request！
