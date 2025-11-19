"""
ByteFormer HuggingFace Migration Training Script
示例运行命令：
# 基础训练（使用全部数据）
python byteformer-hf-migration/scripts/train_hf_byteformer_cls.py --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --num_train_epochs 3 --learning_rate 5e-3 --eval_steps 100 --logging_steps 50 --save_steps 500 --lr_scheduler_type linear --gradient_accumulation_steps 1 --num_train_samples 6000 --num_eval_samples 100 --report_to none

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
from utils.hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
import torchtext
torchtext.disable_torchtext_deprecation_warning()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer HuggingFace Migration Training")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--num_classes", type=int, default=1000, help="分类类别数")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")  
    parser.add_argument("--output_dir", type=str, default="./checkpoints/byteformer_hf_training", help="训练输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="每设备训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="每设备验证批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"], help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--fp16", action="store_true", default=False, help="启用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=False, help="启用BF16混合精度")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点总数限制")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "none", "wandb", "tensorboard"], help="日志报告工具")

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("ByteFormer HuggingFace Migration Training")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"预训练权重: {args.weights}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.per_device_train_batch_size}")
    print("=" * 50)
    
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
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    num_classes = args.num_classes  # 使用命令行参数
    
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    weights = torch.load(args.weights, map_location='cpu')
    model_state = model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    model.model.load_state_dict(pretrained_state, strict=False)
    model.train()
    
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
            # 只输入0-9类别，标签直接用原始MNIST标签
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
        
        collated = byteformer_image_collate_fn(corenet_batch, opts)
        input_ids = collated["samples"]
        labels = torch.stack(labels)
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    # Create datasets
    train_ds = BitstreamDataset(split="train", num_classes=args.num_classes, num_samples=args.num_train_samples)
    eval_ds = BitstreamDataset(split="test", num_classes=args.num_classes, num_samples=args.num_eval_samples)
    
    # Custom Training Arguments - simplified version based on test.py
    training_args = MySeq2SeqTrainingArguments(
        output_dir=args.output_dir,
        train_batch_size=args.per_device_train_batch_size,  # Use original parameter name
        eval_batch_size=args.per_device_eval_batch_size,   # Use original parameter name  
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy=args.evaluation_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=int(args.warmup_ratio * 1000),  # Convert ratio to steps roughly
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to if args.report_to not in [None, "none"] else None,
    )
    
    class ClsTrainer(MySeq2SeqTrainer):
        def evaluate(self, eval_dataset=None, desc="Eval", ignore_keys=None, metric_key_prefix: str = "eval"):
            """正确的评估函数，计算损失和准确率"""
            from torch.utils.data import DataLoader
            
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            self.model.eval()
            device = self.device
            
            total_loss = 0.0
            correct = 0
            total = 0
            num_batches = 0
            debug_samples = []  # 存储前几个样本的预测信息
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_dataset):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # 前向传播计算损失和logits
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
                    
                    # 累积损失
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 计算预测和准确率
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    # 收集前10个样本的调试信息
                    if len(debug_samples) < 10:
                        batch_size = labels.size(0)
                        for i in range(min(batch_size, 10 - len(debug_samples))):
                            pred_class = preds[i].item()
                            true_class = labels[i].item()
                            # 获取预测概率分布（前3个最高概率的类别）
                            probs = torch.softmax(logits[i], dim=-1)
                            top3_probs, top3_indices = torch.topk(probs, k=3)
                            
                            debug_samples.append({
                                'sample_idx': len(debug_samples),
                                'predicted': pred_class,
                                'true_label': true_class,
                                'correct': pred_class == true_class,
                                'top3_classes': top3_indices.cpu().tolist(),
                                'top3_probs': top3_probs.cpu().tolist()
                            })
            
            # 计算平均损失和准确率
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            acc = correct / total if total > 0 else 0.0
            
            # 打印调试信息
            print(f"\n[Custom Eval] Loss: {avg_loss:.4f}, Accuracy: {acc:.4f} ({correct}/{total})")
            print("=" * 60)
            print("前10个样本的预测结果：")
            print("-" * 60)
            for sample in debug_samples:
                status = "✓" if sample['correct'] else "✗"
                print(f"样本 {sample['sample_idx']:2d}: {status} 预测={sample['predicted']}, 真实={sample['true_label']}")
                print(f"         Top3预测: {sample['top3_classes']} (概率: {[f'{p:.3f}' for p in sample['top3_probs']]}")
            print("=" * 60)
            
            self.model.train()

            if self.compute_metrics is not None:
                # 构造metrics字典
                metrics = {
                    "accuracy": acc,
                    "samples": total
                }
                return avg_loss, metrics
            else:
                return avg_loss
    
    # Initialize custom trainer
    trainer = ClsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=bitstream_collate_fn,
        tokenizer=None,  # Not needed for vision tasks
    )
    
    print("开始使用自定义训练框架训练...")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(eval_ds)}")
    print(f"训练参数: {training_args}")
    
    # Start training
    trainer.train()
    
    print("训练完成！")

if __name__ == "__main__":
    main()