#!/usr/bin/env python3
"""InternVL3_5-8B training script with model-specific logic.

Extends train_hf_trainer.py with InternVL-specific data collation, 
image preprocessing, and tokenization following official examples.
Example:
    python tools/train_internvl_hf_trainer.py \
        --folder PureT/experiments/ByteCaption_XE_internvl \
        --dataset coco \
        --model_id InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
        --processor_id InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
        --local_dir InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
        --train_samples 0 \
        --val_samples 5 \
        --eval_steps 200 \
        --best_metric SPICE \
        --early_stop_patience 4 \
        --max_epoch 2 \
        --batch_size 1 \
        --grad_accum_steps 8 \
        --gradient_checkpointing \
        --num_workers 8 \
        --train_max_length 512 \
        --train_truncation 1 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --attn_implementation flash_attention_2 \
        --disable_wandb
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
from torchvision.transforms.functional import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "PureT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

# Import config system
from lib.config import cfg, cfg_from_file

# Import from generic trainer
from tools.train_hf_trainer import (
    CaptionTrainer,
    _load_model_and_processor,
    _select_captions,
    _apply_hf_overrides,
    _apply_dataloader_overrides,
    _apply_solver_overrides,
)
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.datasets_.flickr8k_dataset_hf import Flickr8kDataset
from transformers import AutoTokenizer, TrainingArguments

DEFAULT_MODEL_ID = "InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF"
DEFAULT_LOCAL_DIR = str(PROJECT_ROOT / "InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF")
DEFAULT_FOLDER = "PureT/experiments/ByteCaption_XE_internvl"
DEFAULT_MIRROR = "https://hf-mirror.com"

# system prompt for structured reasoning
SYSTEM_PROMPT = """
You are a vision captioning model.
""".strip()


class InternVLCollator:
    """InternVL-specific data collator using messages format and processor."""
    
    def __init__(
        self,
        processor,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt: str = "Describe this image in detail.",
        label_ignore: int = -100,
        seq_per_img: int = 1,
    ):
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.label_ignore = label_ignore
        self.seq_per_img = max(int(seq_per_img), 1)
    
    def _build_messages(self, caption: Optional[str] = None):
        """Build messages in chat format for InternVL processor.
        
        Returns:
            messages: List of message dicts with role and content
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "<image_placeholder>"},
                    {"type": "text", "text": self.user_prompt},
                ],
            },
        ]
        
        if caption:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            })
            
        return messages
    
    def __call__(self, batch: Sequence[Tuple[Any, ...]]):
        """Collate batch using processor for correct InternVL format."""
        indices, captions_list, _gv_feat, images = zip(*batch)
        
        # Expand images and captions
        expanded_images: List[Any] = []
        expanded_captions: List[str] = []
        for img, caps in zip(images, captions_list):
            selected = _select_captions(list(caps), self.seq_per_img)
            for cap in selected:
                expanded_images.append(img)
                expanded_captions.append(cap)
        
        # Build messages for each sample
        full_messages = [self._build_messages(cap) for cap in expanded_captions]
        question_messages = [self._build_messages(None) for _ in expanded_captions]
        
        # Apply chat template to convert messages to text prompts
        full_prompts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in full_messages
        ]
        question_prompts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in question_messages
        ]
        
        # Use processor to handle image+text pairs with correct formatting
        full_inputs = self.processor(
            images=expanded_images,
            text=full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        question_inputs = self.processor(
            images=expanded_images,
            text=question_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        # Create labels from full_inputs
        labels = full_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Mask padding tokens
        if pad_token_id is not None:
            labels[labels == pad_token_id] = self.label_ignore
        
        # Mask prompt tokens: find where question ends and answer begins
        # Strategy: tokenize prompts separately to find their lengths
        for i in range(labels.shape[0]):
            # Get question token count (this is the prompt that should be masked)
            question_len = question_inputs["input_ids"][i].ne(pad_token_id).sum().item() if pad_token_id is not None else len(question_inputs["input_ids"][i])
            # Mask the first question_len tokens (the prompt part)
            labels[i, :question_len] = self.label_ignore
        
        # Return data compatible with InternVL model
        result = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs.get("attention_mask"),
            "pixel_values": full_inputs.get("pixel_values"),
            "image_flags": full_inputs.get("image_flags"),
            "num_patches_list": full_inputs.get("num_patches_list"),
            "labels": labels,
        }
        
        return result

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="InternVL training with HF Trainer")
    parser.add_argument("--folder", type=str, default=DEFAULT_FOLDER)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--processor_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--local_dir", type=str, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--best_metric", type=str, default="SPICE")
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Training prompts (for compatibility with _apply_hf_overrides)
    parser.add_argument("--train_mode", type=str, default=None)
    parser.add_argument("--train_system_prompt", type=str, default=None)
    parser.add_argument("--train_user_prompt", type=str, default=None)
    parser.add_argument("--train_max_length", type=int, default=512)
    parser.add_argument("--train_truncation", type=int, default=1)
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default=None)
    parser.add_argument("--lora_task_type", type=str, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", 
                       default=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--lora_modules_to_save", nargs="+", default=None)
    parser.add_argument("--lora_save_full_model", action="store_true")
    
    # Data loader parameters (for compatibility)
    parser.add_argument("--seq_per_img", type=int, default=None)
    parser.add_argument("--shuffle", type=int, default=None)
    parser.add_argument("--pin_memory", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    
    # Solver parameters (for compatibility)
    parser.add_argument("--test_interval", type=int, default=None)
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    
    # Model parameters
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--bf16", type=int, default=1, help="Enable bf16 training (1=True, 0=False)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    return parser.parse_args()


def main() -> None:
    """Main training function with InternVL-specific logic."""
    args = parse_args()
    
    # Load config using the same system as train_hf_trainer.py
    config_file = f"config_{args.dataset}.yml"
    config_path = Path(args.folder) / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    cfg_from_file(str(config_path))
    cfg.ROOT_DIR = str(Path(args.folder))
    
    # Apply CLI overrides to config using same functions as train_hf_trainer.py
    _apply_hf_overrides(args)
    _apply_dataloader_overrides(args)
    _apply_solver_overrides(args)
    
    # Additional InternVL-specific config
    if not hasattr(cfg.MODEL, "HF"):
        from yacs.config import CfgNode
        cfg.MODEL.HF = CfgNode()
    
    hf_cfg = cfg.MODEL.HF
    hf_cfg.MODEL_ID = args.model_id
    hf_cfg.PROCESSOR_ID = args.processor_id
    hf_cfg.LOCAL_DIR = args.local_dir
    hf_cfg.TRUST_REMOTE_CODE = True
    
    # Load model and processor using generic loader
    print(f"[InternVL] Loading model from {args.local_dir}")
    model, processor = _load_model_and_processor(hf_cfg)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("[InternVL] Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    
    # Load datasets
    print(f"[InternVL] Loading {args.dataset} dataset")
    train_samples = args.train_samples if args.train_samples > 0 else None
    val_samples = args.val_samples if args.val_samples > 0 else None
    
    if args.dataset == "coco":
        train_dataset = CocoDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.get("INPUT_SEQ_PATH", None),
            target_seq=cfg.DATA_LOADER.get("TARGET_SEQ_PATH", None),
            gv_feat_path=cfg.DATA_LOADER.get("TRAIN_GV_FEAT", ""),
            seq_per_img=cfg.DATA_LOADER.get("SEQ_PER_IMG", 1),
            max_feat_num=cfg.DATA_LOADER.get("MAX_FEAT", -1),
            max_samples=train_samples,
            return_captions=True,
            return_pil=True,
        )
        val_dataset = CocoDataset(
            image_ids_path=cfg.DATA_LOADER.VAL_ID,
            input_seq=None,
            target_seq=None,
            gv_feat_path=cfg.DATA_LOADER.get("VAL_GV_FEAT", ""),
            seq_per_img=1,
            max_feat_num=cfg.DATA_LOADER.get("MAX_FEAT", -1),
            max_samples=val_samples,
            return_captions=True,
            return_pil=True,
        )
    else:
        train_dataset = Flickr8kDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.get("INPUT_SEQ_PATH", None),
            target_seq=cfg.DATA_LOADER.get("TARGET_SEQ_PATH", None),
            gv_feat_path=cfg.DATA_LOADER.get("TRAIN_GV_FEAT", ""),
            seq_per_img=cfg.DATA_LOADER.get("SEQ_PER_IMG", 1),
            max_feat_num=cfg.DATA_LOADER.get("MAX_FEAT", -1),
            max_samples=train_samples,
            return_captions=True,
            return_pil=True,
        )
        val_dataset = Flickr8kDataset(
            image_ids_path=cfg.DATA_LOADER.VAL_ID,
            input_seq=None,
            target_seq=None,
            gv_feat_path=cfg.DATA_LOADER.get("VAL_GV_FEAT", ""),
            seq_per_img=1,
            max_feat_num=cfg.DATA_LOADER.get("MAX_FEAT", -1),
            max_samples=val_samples,
            return_captions=True,
            return_pil=True,
        )
    
    # Get prompts from config (with R1 default if not specified)
    system_prompt = getattr(cfg, "SYSTEM_PROMPT", SYSTEM_PROMPT)
    user_prompt = getattr(cfg, "USER_PROMPT", "Describe this image in detail.")
    
    collator = InternVLCollator(
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        label_ignore=-100,
        seq_per_img=1,
    )
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.folder) / "checkpoints_internvl")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps if args.save_steps else args.eval_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=10,
        bf16=bool(args.bf16),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric.lower(),
        greater_is_better=True,
        report_to="none" if args.disable_wandb else "wandb",
    )
    
    # Create trainer
    trainer = CaptionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
    )
    
    # Train
    print("[InternVL] Starting training")
    trainer.train()
    
    # Best model is already saved by trainer's callback
    print(f"[InternVL] Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()

