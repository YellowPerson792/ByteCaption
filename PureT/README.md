# ByteCaption（ByteFormer + PureT）

从 JPEG 码流直接生成图像描述：利用 CoreNet 中的 ByteFormer 作为字节级编码器，接上 PureT Caption 解码器完成端到端训练与评估。

## 目录速览
- `PureT/main.py`：训练入口（支持 COCO / Flickr8k），默认每个 epoch 评估，也可通过 `--eval_steps` 按步评估。
- `PureT/main_val.py`：仅评估入口，便于在已有 checkpoint 上做离线验证。
- `PureT/datasets_`：数据集封装（`coco_dataset_hf.py` 从 HF `AbdoTW___coco_2014_karpathy` 读取图像；`data_loader_byteformer_coco.py` 将图像送入 ByteFormer 的 collate 流程）。
- `corenet/data/transforms/image_bytes.py`：字节级数据增强（PIL 编码、字节损坏、掩码、置换、噪声等）。
- `corenet/data/collate_fns/byteformer_collate_functions.py`：ByteFormer 专用 collate，负责调用上述增强并对变长字节序列做 padding。
- `PureT/byteformer_immigration.py`：将 CoreNet ByteFormer 封装为 HF `PreTrainedModel`，并提供 `get_opts()` 读取 ByteFormer 配置/权重。
- `PureT/experiments/ByteCaption_XE/`：当前使用的配置与输出目录（`config_coco.yml`、`snapshot/`、`result/` 等）。
- 实用脚本：`PureT/analyze_stream_length.py`（统计 JPEG 码流长度），`PureT/compare_encoding.py`、`encoding_compare_samples/`（编码对比样例）。

## 环境准备
```bash
cd ByteCaption
python -m pip install -r requirements.txt
# 需要额外工具时：python -m pip install -r requirements-optional.txt

# 建议设置 PYTHONPATH，方便直接运行 PureT 入口
export PYTHONPATH=$(pwd)
```

## 数据准备
- HF COCO 数据集：`PureT/data/coco_karpathy/AbdoTW___coco_2014_karpathy/{train,validation,test}`（`coco_dataset_hf.py` 默认读取此路径）。
- 图像 ID 列表：`PureT/data/coco_karpathy/train_ids.json`、`validation_ids.json`（`config_coco.yml` 中指定）。
- 词表：首次运行会在 `PureT/data/coco_karpathy/coco_vocabulary.txt` 自动生成（也可提前准备）。
- 若切换 Flickr8k，请对应修改 `config_flickr8k.yml` 与启动参数 `--dataset flickr8k`。

## 训练示例
```bash
# 单机单卡/多卡都会在 main.py 内自动选择 DataParallel 或 DDP
python PureT/main.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --eval_steps 600 \
  --val_samples 50 \
  --load_weights \
  --freeze_backbone \
  --disable_wandb
# 说明：移除 --disable_wandb 即可开启 wandb；移除 --freeze_backbone 可端到端微调。

# 多卡示例
# torchrun --nproc_per_node=2 --master_port=12355 PureT/main.py --folder ...（其余参数同上）
```

关键行为：
- ByteFormer 配置与预训练权重：在 `PureT/byteformer_immigration.py:get_opts()` 中固定为 `byteformer_hf_migration/configs/conv_kernel_size=4,window_sizes=[128].yaml` 与 `byteformer_hf_migration/weights/imagenet_jpeg_q60_k4_w128.pt`，可按需修改路径或参数。
- 数据流：`CocoDataset` 读取 PIL 图 → `byteformer_collate` 将每张图包装为 CoreNet batch → `byteformer_image_collate_fn` 执行 PIL 编码 (`PILSave`)，然后按配置叠加字节损坏、掩码、置换、噪声，最后 padding 并返回字节序列张量。
- 训练配置：`PureT/experiments/ByteCaption_XE/config_coco.yml` 控制批大小、学习率、保存频率、模型维度等；`MODEL.TYPE` 设为 `PureT_byteformer` 以走字节流分支。

## 验证 / 推理
```bash
# 仅验证，不进入训练循环
python PureT/main_val.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --val_samples 500 \
  --resume -1 \                 # -1 表示加载 best_model.pth；>0 加载指定 epoch
  --disable_wandb
```
评估器将使用 `config_coco.yml` 中的 `INFERENCE.VAL_ANNFILE` 与 `VOCAB`，并自动复用 ByteFormer collate 逻辑（或 BLIP 模式，取决于 `MODEL.TYPE`）。

## 调试与排错提示
- 查看训练日志：`PureT/experiments/ByteCaption_XE/log.txt`；模型快照保存在 `snapshot/`，评估结果在 `result/`。
- 字节流统计：`python PureT/analyze_stream_length.py --dataset_path PureT/data/coco_karpathy/validation_ids.json --max_samples 2000 --quality 95`。
- 字节增强配置：在 `image_bytes.py` 中支持 `--image-augmentation.pil-save.*`、`--image-augmentation.byte-stream-corrupter.*` 等参数（由 `get_opts()` 生成的 CoreNet opts 读取）。需要更多/更少损坏可调整 `level` 或 `types`。
- 训练/验证样本调试：`data_loader_byteformer_coco.py` 的 `blip_collate_val` 会尝试将损坏后码流解码回图像并写入 `evaluation_samples/` 便于检查。
