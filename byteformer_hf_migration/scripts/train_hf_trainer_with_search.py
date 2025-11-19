"""
ByteFormer HuggingFace Trainer Training Script with Hyperparameter Search
基于原始 train_hf_byteformer_cls.py，使用 HF Trainer 实现训练，并集成超参数自动搜索。

示例运行命令：
python byteformer-hf-migration/scripts/train_hf_trainer_with_search.py --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --num_train_epochs 5 --learning_rate 2e-4 --eval_steps 100 --logging_steps 50 --save_steps 500 --lr_scheduler_type linear --gradient_accumulation_steps 1 --num_train_samples 6000 --num_eval_samples 100 --warmup_ratio 0.04 --report_to wandb
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from corenet.data.transforms.image_bytes import PILSave
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
import numpy as np
import torchtext
torchtext.disable_torchtext_deprecation_warning()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer HuggingFace Trainer Training with Hyperparameter Search")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--num_classes", type=int, default=1000, help="分类类别数")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")  
    parser.add_argument("--output_dir", type=str, default="./checkpoints/byteformer_hf_trainer_training", help="训练输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="每设备训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="每设备验证批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam优化器beta1参数")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam优化器beta2参数")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam优化器epsilon参数")
    parser.add_argument("--fp16", action="store_true", default=False, help="启用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=False, help="启用BF16混合精度")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点总数限制")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "none", "wandb", "tensorboard"], help="日志报告工具")
    
    # 超参数搜索相关参数
    parser.add_argument("--hp_search", action="store_true", default=False, help="是否启用超参数搜索")
    parser.add_argument("--n_trials", type=int, default=10, help="超参数搜索试验次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()

# 全局变量，用于在 model_init 中访问参数
global_args = None
global_opts = None

def model_init():
    """模型初始化函数，用于超参数搜索"""
    args = global_args
    opts = global_opts
    
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    
    # 加载预训练权重
    if args.weights and os.path.exists(args.weights):
        weights = torch.load(args.weights, map_location='cpu')
        model_state = model.model.state_dict()
        pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
        model.model.load_state_dict(pretrained_state, strict=False)
        print(f"已加载预训练权重: {args.weights}")
    
    return model

def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def hp_space(trial=None):
    """定义超参数搜索空间，兼容 Optuna(trial) 和 WandB(sweep config) 后端"""
    # WandB sweep 后端会传入 trial=None，需要返回 sweep 配置
    if trial is None:
        return {
            "method": "bayes",
            "metric": {
                "name": "eval/accuracy",
                "goal": "maximize"
            },
            "parameters": {
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 5e-4
                },
                "per_device_train_batch_size": {
                    "distribution": "categorical",
                    "values": [8, 16, 32]
                },
                "num_train_epochs": {
                    "distribution": "int_uniform",
                    "min": 2,
                    "max": 5
                },
                "warmup_ratio": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.2
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-4,
                    "max": 1e-1
                },
                "lr_scheduler_type": {
                    "distribution": "categorical",
                    "values": ["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"]
                },
                # "adam_beta1": {
                #     "distribution": "uniform",
                #     "min": 0.85,
                #     "max": 0.95
                # },
                # "adam_beta2": {
                #     "distribution": "uniform",
                #     "min": 0.99,
                #     "max": 0.999
                # }
            }
        }
    # Optuna backend 使用 trial 对象建议
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "warmup_ratio": trial.suggest_uniform("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 1e-1),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"]),
        "adam_beta1": trial.suggest_uniform("adam_beta1", 0.85, 0.95),
        "adam_beta2": trial.suggest_uniform("adam_beta2", 0.99, 0.999),
    }

def main():
    global global_args, global_opts
    
    # Parse command line arguments
    args = parse_args()
    global_args = args
    
    set_seed(args.seed)
    
    print("ByteFormer HuggingFace Trainer Training with Hyperparameter Search")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print(f"预训练权重: {args.weights}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.per_device_train_batch_size}")
    print(f"超参数搜索: {args.hp_search}")
    if args.hp_search:
        print(f"搜索试验次数: {args.n_trials}")
    print("=" * 70)
    
    # Load CoreNet configuration using args list (original way)
    corenet_args = [
        "--common.config-file", args.config,
        "--model.classification.pretrained", args.weights,
        "--model.classification.n-classes", str(args.num_classes),
        "--dataset.root-train", "./mnist_data",
        "--dataset.root-val", "./mnist_data",
        "--common.accum-freq", str(args.gradient_accumulation_steps),
        "--common.log-freq", str(args.logging_steps),
    ]
    opts = get_training_arguments(args=corenet_args)
    global_opts = opts
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    num_classes = args.num_classes  # 使用命令行参数
    
    # 图像预处理：将PIL图像转换为tensor格式以适配PILSave
    def pil_to_tensor_transform(img):
        """将PIL图像转换为CoreNet期望的tensor格式"""
        # 先转为torch tensor [C, H, W]，值范围[0, 1]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 标准ImageNet尺寸
            transforms.ToTensor()  # 转为[0,1]范围的tensor
        ])
        return transform(img)

    # Updated Unified Dataset Class with proper HF integration
    class BitstreamDataset(torch.utils.data.Dataset):
        def __init__(self, split="train", num_classes=1000, num_samples=None):
            self.dataset = load_dataset("uoft-cs/cifar10", split=split)
            self.num_classes = num_classes

            if num_samples is not None and num_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(num_samples))
                print(f"使用 {split} 数据集的前 {num_samples} 个样本")
            else:
                print(f"使用完整的 {split} 数据集，共 {len(self.dataset)} 个样本")
            
        def __getitem__(self, idx):
            example = self.dataset[idx]
            img = example["img"]  # PIL Image
            original_label = example["label"]
            img = img.convert('RGB')
            img_tensor = pil_to_tensor_transform(img)
            # 只输入0-9类别，标签直接用原始CIFAR标签
            return {
                'input_ids': img_tensor,
                'labels': torch.tensor(original_label, dtype=torch.long)
            }
            
        def __len__(self):
            return len(self.dataset)

    def bitstream_collate_fn(batch):
        """Custom collate function for HF trainer compatibility"""
        images = []
        labels = []
        for item in batch:
            images.append(item['input_ids'])
            labels.append(item['labels'])

        corenet_batch = []
        for img_tensor, label in zip(images, labels):
            corenet_batch.append({"samples": img_tensor, "targets": label})
        
        collated = byteformer_image_collate_fn(corenet_batch, global_opts)
        input_ids = collated["samples"]
        labels = torch.stack(labels)
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    # Create datasets
    train_ds = BitstreamDataset(split="train", num_classes=args.num_classes, num_samples=args.num_train_samples)
    eval_ds = BitstreamDataset(split="test", num_classes=args.num_classes, num_samples=args.num_eval_samples)
    
    # HuggingFace Training Arguments (原 MySeq2SeqTrainingArguments 参数映射)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to if args.report_to not in [None, "none"] else None,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,  # 保留自定义列
    )

    # 创建初始模型（如果不进行超参数搜索）
    if not args.hp_search:
        model = model_init()
    else:
        model = None  # 超参数搜索时不需要初始模型

    trainer = Trainer(
        model=model,
        model_init=model_init if args.hp_search else None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=bitstream_collate_fn,
        compute_metrics=compute_metrics,
    )

    print("开始使用 HuggingFace Trainer 训练...")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(eval_ds)}")
    print(f"训练参数: {training_args}")
    
    if args.hp_search:
        print(f"\n开始超参数搜索，试验次数: {args.n_trials}")
        # 选择超参数搜索后端
        backend = "wandb" if args.report_to == "wandb" else "optuna"
        print(f"使用 {backend} 后端进行超参数搜索")
        # 超参数搜索
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hp_space,
            n_trials=args.n_trials,
            backend=backend
        )
        print("\n最优超参数:")
        for key, value in best_run.hyperparameters.items():
            print(f"  {key}: {value}")
        print(f"最优分数: {best_run.objective}")
        
        # 用最优参数重新训练
        print("\n使用最优参数进行最终训练...")
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
    
    # Start training
    trainer.train()
    trainer.save_model()
    
    print("训练完成！")

if __name__ == "__main__":
    main()
