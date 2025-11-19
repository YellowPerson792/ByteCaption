"""
ByteFormer + T5 Caption Training Script (Restructured)
使用ByteFormer作为特征提取器，将输出送入完整的T5模型进行图像描述生成

架构设计：
1. ByteFormer：提取图像特征并转换为token序列
2. Feature Projection：将ByteFormer特征投影到T5的embedding空间  
3. T5 Complete Model：使用投影后的特征作为encoder_outputs进行生成

示例运行命令：
python byteformer-hf-migration/scripts/train_byteformer_t5_caption.py --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --learning_rate 5e-5 --eval_steps 10 --logging_steps 50 --save_steps 600 --lr_scheduler_type cosine --gradient_accumulation_steps 2 --report_to none --max_caption_length 16 --num_eval_samples 50 --fp16
"""

import os
# 设置环境变量以避免tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
from functools import partial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Union
from tqdm import tqdm
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from corenet.data.transforms.image_bytes import PILSave
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from utils.hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

CONFIG_MAPPING.register("byteformer", CorenetToHFPretrainedConfig)
MODEL_MAPPING.register(CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel)


class CustomSeq2SeqTrainer(MySeq2SeqTrainer):
    """自定义训练器，重写evaluate方法以支持T5模型的生成评估"""
    
    def evaluate(self, eval_dataset=None, desc="评估中"):
        """重写evaluate方法，支持无参数调用和生成评估"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            print("没有评估数据集，跳过评估")
            return {}
        
        print(f"开始评估，共 {len(eval_dataset)} 个批次...")
        
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataset, desc="评估中", ncols=100)):
                try:
                    # 移动数据到设备
                    input_ids = batch['input_ids'].to(self.device)
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                    decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 打印张量形状进行调试
                    if i == 0:  # 只在第一次打印
                        print(f"Debug - input_ids shape: {input_ids.shape}")
                        print(f"Debug - decoder_input_ids shape: {decoder_input_ids.shape}")
                        print(f"Debug - decoder_attention_mask shape: {decoder_attention_mask.shape}")
                        print(f"Debug - labels shape: {labels.shape}")
                    
                    # 计算loss - 使用较小的批次避免内存问题
                    batch_size = input_ids.size(0)
                    if batch_size > 2:  # 如果批次太大，分批处理
                        batch_losses = []
                        for start_idx in range(0, batch_size, 1):  # 每次处理1个样本
                            end_idx = min(start_idx + 1, batch_size)
                            mini_batch_outputs = self.model(
                                input_ids=input_ids[start_idx:end_idx],
                                decoder_input_ids=decoder_input_ids[start_idx:end_idx],
                                decoder_attention_mask=decoder_attention_mask[start_idx:end_idx],
                                labels=labels[start_idx:end_idx]
                            )
                            batch_losses.append(mini_batch_outputs.loss.item())
                        avg_batch_loss = sum(batch_losses) / len(batch_losses)
                        total_loss += avg_batch_loss
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=labels
                        )
                        total_loss += outputs.loss.item()
                    
                    # 生成预测 - 逐个处理避免内存问题
                    for single_idx in range(input_ids.size(0)):
                        single_input = input_ids[single_idx:single_idx+1]
                        single_label = labels[single_idx:single_idx+1]
                        
                        generated_ids = self.model.generate(
                            input_ids=single_input,
                            max_length=getattr(self.args, 'generation_max_length', 50),
                            num_beams=getattr(self.args, 'generation_num_beams', 4),
                            early_stopping=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        
                        predictions.extend(generated_ids.cpu().tolist())
                        references.extend(single_label.cpu().tolist())
                
                except RuntimeError as e:
                    print(f"Error in evaluation batch {i}: {e}")
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # 定期清理内存
                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.model.train()
        
        # 计算平均损失
        avg_loss = total_loss / len(eval_dataset)
        result = {'eval_loss': avg_loss}
        
        # 计算指标
        if self.compute_metrics and predictions:
            # 创建类似HF的评估预测对象
            eval_pred = type('EvalPrediction', (), {})()
            eval_pred.predictions = np.array(predictions)
            eval_pred.label_ids = np.array(references)
            
            metrics = self.compute_metrics(eval_pred)
            for key, value in metrics.items():
                result[f'eval_{key}'] = value
        
        return result


class ByteFormerT5Model(PreTrainedModel):
    """
    ByteFormer + T5 完整模型
    
    架构：
    1. ByteFormer提取视觉特征
    2. 特征投影层将ByteFormer特征映射到T5的embedding空间
    3. T5模型将投影后的特征作为encoder_outputs进行文本生成
    """
    
    def __init__(self, byteformer_model, t5_model, config):
        super().__init__(config)
        self.byteformer = byteformer_model
        self.t5 = t5_model
        self.config = config
        
        # 获取维度信息
        # 从config中获取ByteFormer的hidden_size，避免直接访问模型内部结构
        self.byteformer_dim = config.hidden_size  # ByteFormer输出维度
        self.t5_dim = t5_model.config.d_model  # T5的embedding维度
        
        # 特征投影层：将ByteFormer特征投影到T5的embedding空间
        self.feature_projection = nn.Linear(self.byteformer_dim, self.t5_dim)
        
        # 初始化投影层
        nn.init.normal_(self.feature_projection.weight, std=0.02)
        nn.init.zeros_(self.feature_projection.bias)
        
        print(f"ByteFormer dimension: {self.byteformer_dim}")
        print(f"T5 dimension: {self.t5_dim}")
        
    def extract_byteformer_features(self, input_ids):
        """
        使用ByteFormer提取视觉特征
        
        Args:
            input_ids: ByteFormer的输入token序列 [batch_size, seq_len]
            
        Returns:
            features: 提取的特征 [batch_size, seq_len, byteformer_dim]
            attention_mask: 对应的注意力掩码 [batch_size, seq_len]
        """
        # 步骤1: 获取backbone输入 (embeddings + positional embeddings)
        x, key_padding_mask = self.byteformer.model.get_backbone_inputs(input_ids)
        
        # 步骤2: 通过transformer backbone
        features, updated_mask = self.byteformer.model.backbone_forward(x, key_padding_mask)
        
        # 步骤3: 生成注意力掩码（非padding部分为1，padding部分为0）
        # updated_mask中，-inf表示被mask的位置，其他位置是0
        # 转换为attention_mask: 1表示有效位置，0表示mask位置
        if updated_mask is not None:
            attention_mask = (updated_mask != float("-inf")).float()
        else:
            # 如果没有掩码，认为所有位置都是有效的
            attention_mask = torch.ones(features.shape[:2], dtype=torch.float32, device=features.device)
        
        return features, attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # ByteFormer输入
        attention_mask: Optional[torch.FloatTensor] = None,  # 未使用，ByteFormer内部处理
        decoder_input_ids: Optional[torch.LongTensor] = None,  # T5 decoder输入
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # T5 decoder掩码
        labels: Optional[torch.LongTensor] = None,  # 标签
        **kwargs
    ):
        """
        前向传播
        
        Args:
            input_ids: ByteFormer的输入图像token [batch_size, img_seq_len]
            decoder_input_ids: T5 decoder的输入token [batch_size, text_seq_len]
            decoder_attention_mask: T5 decoder的注意力掩码
            labels: 用于计算loss的标签
        """
        # 步骤1: 使用ByteFormer提取视觉特征
        byteformer_features, byteformer_attention_mask = self.extract_byteformer_features(input_ids)
        
        # 步骤2: 将ByteFormer特征投影到T5的embedding空间
        projected_features = self.feature_projection(byteformer_features)
        
        # 步骤3: 构造encoder_outputs给T5使用
        encoder_outputs = BaseModelOutput(
            last_hidden_state=projected_features,
            hidden_states=None,
            attentions=None
        )
        
        # 步骤4: 使用T5进行生成，将投影后的特征作为encoder输出
        t5_outputs = self.t5(
            encoder_outputs=encoder_outputs,
            # encoder_attention_mask=byteformer_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        
        return t5_outputs
    
    def generate(self, input_ids, **generation_kwargs):
        """
        生成文本
        
        Args:
            input_ids: ByteFormer的输入图像token
            **generation_kwargs: T5生成的参数
        """
        # 提取ByteFormer特征
        byteformer_features, byteformer_attention_mask = self.extract_byteformer_features(input_ids)
        
        # 投影到T5空间
        projected_features = self.feature_projection(byteformer_features)
        
        # 构造encoder_outputs
        encoder_outputs = BaseModelOutput(
            last_hidden_state=projected_features,
            hidden_states=None,
            attentions=None
        )
        
        # 使用T5生成
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=byteformer_attention_mask,
            **generation_kwargs
        )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer + T5 Caption Training (Restructured)")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--t5_model", type=str, default="t5-small", help="T5模型名称")
    parser.add_argument("--dataset_name", type=str, default="jxie/flickr8k", help="数据集名称")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")
    parser.add_argument("--max_caption_length", type=int, default=50, help="最大caption长度")
    parser.add_argument("--max_byteformer_length", type=int, default=2048, help="ByteFormer最大输入长度")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/byteformer_t5_caption_v2", help="训练输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每设备训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每设备验证批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"], help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="预热比例")
    parser.add_argument("--fp16", action="store_true", default=False, help="启用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=False, help="启用BF16混合精度")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点总数限制")
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "none", "wandb", "tensorboard"], help="日志报告工具")
    return parser.parse_args()


# 图像预处理
def pil_to_tensor_transform(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img)

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", num_samples=None, dataset_name="jxie/flickr8k"):
        # 直接使用datasets对象，不预先加载所有数据
        self.dataset = load_dataset(dataset_name, split=split)
        self.dataset_name = dataset_name
        self.split = split
        self.total_samples = len(self.dataset) * 5
        if num_samples is not None and num_samples < self.total_samples:
            self.total_samples = num_samples
            print(f"使用 {split} 数据集的前 {num_samples} 个样本")
        else:
            print(f"使用完整的 {split} 数据集，共 {self.total_samples} 个样本")
            
    def __getitem__(self, idx):
        # 动态计算对应的数据集索引和caption索引
        dataset_idx = idx // 5  # 每个图片对应5个caption
        caption_idx = idx % 5   # caption编号 0-4
        item = self.dataset[dataset_idx]
        img = item["image"] if "image" in item else item["jpg"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = pil_to_tensor_transform(img)
        caption = item.get(f"caption_{caption_idx}", "")
        return {
            'image_tensor': img_tensor,
            'caption': caption
        }
        
    def __len__(self):
        return self.total_samples


def caption_collate_fn(batch, tokenizer, opts, max_caption_length=50):
    """自定义批处理函数，处理数据格式"""
    images = []
    captions = [] 
    for item in batch:
        images.append(item['image_tensor'])
        captions.append(item['caption'])
        
    # 将tensor图像转换为ByteFormer输入格式
    corenet_batch = []
    for img_tensor in images:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    input_ids = collated["samples"]
    
    # 处理文本数据 - 为T5格式化（添加前缀）
    t5_captions = [f"caption: {caption}" for caption in captions]
    caption_tokens = tokenizer(
        t5_captions,
        padding='longest',
        max_length=max_caption_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # 为T5创建decoder_input_ids和labels
    labels = caption_tokens.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # T5 decoder输入需要向右偏移
    decoder_input_ids = torch.full_like(labels, tokenizer.pad_token_id)
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = tokenizer.pad_token_id  # T5使用pad_token作为开始token
    
    return {
        "input_ids": input_ids,  # ByteFormer输入
        "decoder_input_ids": decoder_input_ids,  # T5 decoder输入
        "decoder_attention_mask": caption_tokens.attention_mask,  # T5 decoder mask
        "labels": labels,  # 用于计算loss
    }


def compute_metrics(eval_pred, tokenizer):
    """计算评估指标"""
    labels_ids = eval_pred.label_ids
    pred_ids = eval_pred.predictions
    
    # Convert to numpy arrays if needed
    if not isinstance(labels_ids, np.ndarray):
        labels_ids = np.array(labels_ids)
    if not isinstance(pred_ids, np.ndarray):
        pred_ids = np.array(pred_ids)
    
    # 解码预测结果
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    # 处理标签：将-100替换为pad_token_id以便解码
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels_ids = np.where(labels_ids != -100, labels_ids, pad_token_id)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    # 计算BLEU分数
    bleu_1_scores = []
    bleu_4_scores = []
    
    for ref, pred in zip(label_str, pred_str):
        # 分词
        reference = [nltk.word_tokenize(ref.lower())]
        candidate = nltk.word_tokenize(pred.lower())
        
        # 计算BLEU分数
        smoothing_function = SmoothingFunction().method4
        if len(candidate) > 0 and len(reference[0]) > 0:
            try:
                bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
                bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
                bleu_1_scores.append(bleu_1)
                bleu_4_scores.append(bleu_4)
            except:
                bleu_1_scores.append(0.0)
                bleu_4_scores.append(0.0)
        else:
            bleu_1_scores.append(0.0)
            bleu_4_scores.append(0.0)
    
    results = {
        'bleu1': round(np.mean(bleu_1_scores), 4),
        'bleu4': round(np.mean(bleu_4_scores), 4),
        'generation_length': round(np.mean([len(pred.split()) for pred in pred_str]), 4)
    }
    
    # Print up to 5 predictions and labels for debugging
    for i, (ref, pred) in enumerate(zip(label_str, pred_str)):
        if i in [0, 5, 10, 15, 20, 25]:  
            print(f"Sample {i + 1}:")
            print(f"  Reference: {ref}")
            print(f"  Prediction: {pred}\n")
    
    return results

def main():
    args = parse_args()
    
    print("ByteFormer + T5 Caption Training (Restructured)")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"预训练权重: {args.weights}")
    print(f"T5模型: {args.t5_model}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.per_device_train_batch_size}")
    print(f"数据集: {args.dataset_name}")
    print("=" * 60)
    
    # 加载CoreNet配置和ByteFormer模型
    corenet_args = [
        "--common.config-file", args.config,
        "--model.classification.pretrained", args.weights,
        "--model.classification.n-classes", "1000",  # 用于加载预训练权重
        "--dataset.root-train", "./data",
        "--dataset.root-val", "./data",
        "--common.accum-freq", str(args.gradient_accumulation_steps),
        "--common.log-freq", str(args.logging_steps),
    ]
    opts = get_training_arguments(args=corenet_args)
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    
    # 创建ByteFormer model
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    
    # 加载预训练权重
    weights = torch.load(args.weights, map_location='cpu')
    model_state = byteformer_model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    byteformer_model.model.load_state_dict(pretrained_state, strict=False)
    print(f"加载了 {len(pretrained_state)}/{len(model_state)} 个预训练权重")
    
    # 加载T5模型
    t5_config = T5Config.from_pretrained(args.t5_model)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model, config=t5_config)
    
    # 创建tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    
    # 创建组合模型
    model = ByteFormerT5Model(byteformer_model, t5_model, hf_config)
    
    # 准备数据集
    print("准备数据集...")
    train_dataset = CaptionDataset(split="train", num_samples=args.num_train_samples, dataset_name=args.dataset_name)
    eval_dataset = CaptionDataset(split="test", num_samples=args.num_eval_samples, dataset_name=args.dataset_name)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    print("模型结构:")
    print(model)
    
    # 创建collate函数的偏函数，绑定tokenizer和opts
    collate_fn = partial(caption_collate_fn, tokenizer=tokenizer, opts=opts, max_caption_length=args.max_caption_length)
    
    # 计算总训练步数，实现真正的 warmup_ratio
    train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    num_train_samples = len(train_dataset)
    steps_per_epoch = (num_train_samples + train_batch_size - 1) // train_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 配置训练参数
    training_args = MySeq2SeqTrainingArguments(
        output_dir=args.output_dir,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy=args.evaluation_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to if args.report_to not in [None, "none"] else None,
    )
    print(f"训练参数: {training_args}") 
    
    # 创建trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,  # 使用自定义的collate函数
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    
    print("训练完成！")
    
    # 评估模型
    print("开始最终评估...")
    eval_result = trainer.evaluate()
    print("评估结果:")
    for key, value in eval_result.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n生成样例caption...")
    model.eval()
    for i in range(min(3, len(eval_dataset))):
        try:
            sample = eval_dataset[i]
            img_tensor = sample['image_tensor']
            true_caption = sample['caption']
            
            # Process through pipeline
            corenet_item = {"samples": img_tensor, "targets": torch.tensor(0)}
            
            collated = byteformer_image_collate_fn([corenet_item], opts)
            input_ids = collated["samples"].unsqueeze(0)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_length=args.max_caption_length,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"\n样例 {i+1}:")
            print(f"真实caption: {true_caption}")
            print(f"生成caption: {generated_caption}")
            
        except Exception as e:
            print(f"生成样例 {i+1} 时出错: {e}")


if __name__ == "__main__":
    main()
