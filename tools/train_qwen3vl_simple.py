#!/usr/bin/env python3
"""简化版 Qwen3-VL 训练脚本 - 只使用 Transformers 默认配置

NOTE: Prefer `tools/train_hf_trainer.py --use_hf_defaults` for the unified
LoRA training + eval + corruption pipeline; this script is kept for quick runs.

Example usage:
    python tools/train_qwen3vl_simple.py \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_epochs 3 \
        --learning_rate 1e-4 \
        --eval_steps 50 \
        --val_samples 20 \
        --bf16 \
        --max_length 512 \
        --disable_wandb
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
    Qwen3VLForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

from lib.config import cfg, cfg_from_file
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.evaluation.evaler_coco import CocoEvaler


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified Qwen3-VL training")
    
    # 基本配置
    parser.add_argument("--config", type=str, default="PureT/experiments/ByteCaption_XE_qwen/config_coco.yml")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3vl_lora")
    parser.add_argument("--model_path", type=str, default="./Qwen3-VL-8B-Instruct")
    
    # 数据
    parser.add_argument("--train_samples", type=int, default=0, help="0=all")
    parser.add_argument("--val_samples", type=int, default=50)
    
    # 训练超参数（使用 Transformers 默认值）
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # 训练序列长度
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for training")
    parser.add_argument("--truncation", action="store_true", help="Enable truncation")
    
    # 生成参数
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    
    # 评估
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    # 其他
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--disable_wandb", action="store_true")
    
    return parser.parse_args()


class SimpleCollator:
    """简化的 data collator，使用与原配置一致的 prompt"""
    
    def __init__(self, processor, max_length=512, truncation=False, system_prompt="", user_prompt=""):
        self.processor = processor
        self.max_length = max_length
        self.truncation = truncation
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.system_prompt = system_prompt or "You are a vision captioning model."
        self.user_prompt = user_prompt or "You are given a possibly corrupted JPEG image. Output a short COCO-style caption. Use 5-12 words. Output only the caption with no extra text."
    
    def __call__(self, batch):
        images = []
        captions = []
        
        for idx, caps, _, image in batch:
            images.append(image)
            captions.append(caps[0] if caps else "")
        
        # 构建消息（使用与原配置一致的 prompt）
        messages_list = []
        for image, caption in zip(images, captions):
            messages = []
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                })
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {"type": "image", "image": image},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            })
            messages_list.append(messages)
        
        # 处理输入
        try:
            full_inputs = self.processor.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=self.truncation,
                max_length=self.max_length if self.truncation else None,
            )
        except ValueError as e:
            # 如果遇到图像标记不匹配，禁用截断重试
            if "image token count" in str(e).lower():
                full_inputs = self.processor.apply_chat_template(
                    messages_list,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                    truncation=False,
                )
            else:
                raise
        
        # 准备 prompt mask（不包括 assistant 回复）
        prompt_messages_list = [msgs[:-1] for msgs in messages_list]
        try:
            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages_list,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=self.truncation,
                max_length=self.max_length if self.truncation else None,
            )
        except ValueError as e:
            if "image token count" in str(e).lower():
                prompt_inputs = self.processor.apply_chat_template(
                    prompt_messages_list,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                    truncation=False,
                )
            else:
                raise
        
        # 创建 labels
        input_ids = full_inputs['input_ids']
        labels = input_ids.clone()
        
        # Mask padding
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100
        
        # Mask prompt（保留 assistant 回复部分用于训练）
        attention_mask = full_inputs['attention_mask']
        prompt_mask = prompt_inputs['attention_mask']
        
        for i in range(len(labels)):
            prompt_len = prompt_mask[i].sum().item()
            full_len = attention_mask[i].sum().item()
            seq_len = labels.size(1)
            pad_len = seq_len - full_len
            
            # 假设 padding_side='left'
            start = pad_len
            end = min(seq_len, start + prompt_len)
            labels[i, start:end] = -100
        
        batch_dict = {
            'input_ids': full_inputs['input_ids'],
            'attention_mask': full_inputs['attention_mask'],
            'pixel_values': full_inputs['pixel_values'],
            'image_grid_thw': full_inputs['image_grid_thw'],
            'labels': labels,
        }
        
        return batch_dict


class SimpleDecoder:
    """简化的解码器，使用与原配置一致的 prompt"""
    
    def __init__(self, model, processor, num_beams=3, max_new_tokens=30, 
                 system_prompt="", user_prompt=""):
        self.model = model
        self.processor = processor
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt or "You are a vision captioning model."
        self.user_prompt = user_prompt or "You are given a possibly corrupted JPEG image. Output a short COCO-style caption. Use 5-12 words. Output only the caption with no extra text."
    
    def eval(self):
        self.model.eval()
        return self
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def decode_beam(self, **kwargs):
        images = kwargs.get('att_feats', [])
        
        messages_list = []
        for image in images:
            if image is None:
                continue
            messages = []
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {"type": "image", "image": image},
                ],
            })
            messages_list.append(messages)
        
        if not messages_list:
            return ["" for _ in images], None
        
        inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        
        # 移除 prompt 部分
        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = generated_ids[:, prompt_len:]
        
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return captions, None


class SimpleTrainer(Trainer):
    """简化的 Trainer，集成评估"""
    
    def __init__(self, *args, evaler=None, decoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaler = evaler
        self.decoder = decoder
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.evaler is None or self.decoder is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 运行 caption 评估
        metrics = self.evaler(self.decoder, "val")
        prefixed = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        
        self.log(prefixed)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, prefixed
        )
        
        return prefixed


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 加载配置（可选，主要用于数据路径）
    if os.path.exists(args.config):
        cfg_from_file(args.config)
    
    print(f"Loading model from {args.model_path}")
    
    # 加载模型和 processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 应用 LoRA
    from peft import LoraConfig, TaskType, get_peft_model
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 启用梯度检查点（如果需要）
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # 准备数据
    train_samples = args.train_samples if args.train_samples > 0 else None
    val_samples = args.val_samples if args.val_samples > 0 else None
    
    train_set = CocoDataset(
        image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
        input_seq=None,
        target_seq=None,
        gv_feat_path='',
        seq_per_img=5,
        max_feat_num=-1,
        max_samples=train_samples,
        return_captions=True,
        return_pil=True,
    )
    
    val_set = CocoDataset(
        image_ids_path=cfg.DATA_LOADER.VAL_ID,
        input_seq=None,
        target_seq=None,
        gv_feat_path='',
        seq_per_img=1,
        max_feat_num=-1,
        max_samples=val_samples,
        return_captions=True,
        return_pil=True,
    )
    
    evaler = CocoEvaler(
        cfg.DATA_LOADER.VAL_ID,
        '',
        '',
        None,
        max_samples=val_samples,
    )
    
    # Data collator 和 decoder（使用与原配置一致的 prompt）
    system_prompt = "You are a vision captioning model."
    user_prompt = "You are given a possibly corrupted JPEG image. Output a short COCO-style caption. Use 5-12 words. Output only the caption with no extra text."
    
    collator = SimpleCollator(
        processor, 
        max_length=args.max_length,
        truncation=args.truncation,
        system_prompt=system_prompt, 
        user_prompt=user_prompt
    )
    decoder = SimpleDecoder(
        model, 
        processor, 
        num_beams=args.num_beams, 
        max_new_tokens=args.max_new_tokens,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_CIDEr",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[] if args.disable_wandb else ["wandb"],
        seed=args.seed,
    )
    
    # 创建 Trainer
    trainer = SimpleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        tokenizer=processor,
        evaler=evaler,
        decoder=decoder,
    )
    
    # 训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Training complete! Model saved to {final_dir}")


if __name__ == "__main__":
    main()
