#!/usr/bin/env python3
"""InternVL3_5-8B training script with model-specific logic.

Extends train_hf_trainer.py with InternVL-specific data collation, 
image preprocessing, and tokenization following official examples.
Example:
    python tools/train_internvl_hf_trainer.py \
        --folder PureT/experiments/ByteCaption_XE_internvl \
        --dataset coco \
        --model_id OpenGVLab/InternVL3_5-8B \
        --processor_id OpenGVLab/InternVL3_5-8B \
        --local_dir ./InternVL3_5-8B \
        --train_samples 0 \
        --val_samples 5 \
        --eval_steps 200 \
        --best_metric SPICE \
        --early_stop_patience 4 \
        --max_epoch 2 \
        --batch_size 1 \
        --grad_accum_steps 8 \
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
from contextlib import contextmanager
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

# Import from generic trainer
from tools.train_hf_trainer import (
    CaptionTrainer,
    _load_model_and_processor,
    _select_captions,
)
from PureT.datasets_.coco_dataset_hf import CocoDataset
from PureT.datasets_.flickr8k_dataset_hf import Flickr8kDataset
from transformers import AutoTokenizer, TrainingArguments
from omegaconf import OmegaConf

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-8B"
DEFAULT_LOCAL_DIR = str(PROJECT_ROOT / "InternVL3_5-8B")
DEFAULT_FOLDER = "PureT/experiments/ByteCaption_XE_internvl"
DEFAULT_MIRROR = "https://hf-mirror.com"
BACKUP_ROOT = "/root/autodl-fs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """Build InternVL image transform pipeline."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Preprocess image with dynamic tiling for InternVL."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # Add thumbnail if using multi-tile
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image_internvl(image, input_size=448, max_num=12):
    """Load and preprocess image for InternVL."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        # Assume it's already a PIL Image or can be converted
        image = Image.fromarray(np.array(image)).convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLCollator:
    """InternVL-specific data collator using official preprocessing."""
    
    def __init__(
        self,
        tokenizer,
        model_path: str,
        system_prompt: str = "",
        user_prompt: str = "",
        input_size: int = 448,
        max_num_tiles: int = 12,
        label_ignore: int = -100,
        seq_per_img: int = 1,
    ):
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.input_size = input_size
        self.max_num_tiles = max_num_tiles
        self.label_ignore = label_ignore
        self.seq_per_img = max(int(seq_per_img), 1)
    
    def _build_prompt(self, caption: Optional[str] = None) -> str:
        """Build prompt text for InternVL."""
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt.strip())
        if self.user_prompt:
            parts.append(self.user_prompt.strip())
        
        # Add image placeholder
        prompt = "\n".join(parts) if parts else ""
        if prompt:
            prompt = f"<image>\n{prompt}"
        else:
            prompt = "<image>"
        
        if caption is not None:
            prompt = f"{prompt}\n{caption}"
        
        return prompt
    
    def __call__(self, batch: Sequence[Tuple[Any, ...]]):
        """Collate batch using InternVL preprocessing."""
        indices, captions_list, _gv_feat, images = zip(*batch)
        
        # Expand images and captions
        expanded_images: List[Any] = []
        expanded_captions: List[str] = []
        for img, caps in zip(images, captions_list):
            selected = _select_captions(list(caps), self.seq_per_img)
            for cap in selected:
                expanded_images.append(img)
                expanded_captions.append(cap)
        
        # Preprocess images using InternVL pipeline
        all_pixel_values = []
        num_patches_list = []
        for img in expanded_images:
            pixel_values = load_image_internvl(
                img, 
                input_size=self.input_size, 
                max_num=self.max_num_tiles
            )
            all_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        
        # Stack all pixel values
        pixel_values_batch = torch.cat(all_pixel_values, dim=0)
        
        # Build prompts
        full_prompts = [self._build_prompt(cap) for cap in expanded_captions]
        question_prompts = [self._build_prompt(None) for _ in expanded_captions]
        
        # Tokenize
        full_inputs = self.tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        question_inputs = self.tokenizer(
            question_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Create labels by masking prompt tokens
        labels = full_inputs["input_ids"].clone()
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = self.label_ignore
        
        # Mask prompt part in labels
        prompt_lengths = question_inputs["attention_mask"].sum(dim=1)
        for i, plen in enumerate(prompt_lengths):
            if plen > 0:
                labels[i, :plen] = self.label_ignore
        
        return {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "pixel_values": pixel_values_batch,
            "labels": labels,
        }

@contextmanager
def _hf_env(mirror: Optional[str], disable_proxy: bool):
    """Context manager for HuggingFace environment variables."""
    updates = {}
    if mirror:
        updates["HF_ENDPOINT"] = mirror
    if disable_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            updates.setdefault(key, "")

    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _snapshot_ready(path: str) -> bool:
    """Check if model snapshot is complete."""
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_weights = False
    for fname in os.listdir(path):
        if fname.startswith("pytorch_model") or fname.endswith(".safetensors"):
            has_weights = True
            break
    return has_config and has_weights


def _copy_tree(src: str, dst: str) -> None:
    """Recursively copy directory tree."""
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dest_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(dest_root, exist_ok=True)
        for fname in files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dest_root, fname)
            if os.path.exists(dst_path):
                try:
                    if os.path.getsize(dst_path) == os.path.getsize(src_path):
                        continue
                except OSError:
                    pass
            shutil.copy2(src_path, dst_path)


def _ensure_weights(model_id: str, local_dir: str) -> None:
    """Download and backup model weights."""
    if not local_dir:
        return
    if not _snapshot_ready(local_dir):
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            print(f"[InternVL] huggingface_hub unavailable: {exc}")
            return

        os.makedirs(local_dir, exist_ok=True)
        print(f"[InternVL] Downloading {model_id} to {local_dir}")
        with _hf_env(DEFAULT_MIRROR, True):
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )

    backup_name = os.path.basename(os.path.normpath(local_dir))
    if not backup_name:
        return
    backup_dir = os.path.join(BACKUP_ROOT, backup_name)
    if _snapshot_ready(backup_dir):
        return
    os.makedirs(backup_dir, exist_ok=True)
    print(f"[InternVL] Backing up weights to {backup_dir}")
    _copy_tree(local_dir, backup_dir)


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
    parser.add_argument("--train_max_length", type=int, default=512)
    parser.add_argument("--train_truncation", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", 
                       default=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    return parser.parse_args()


def main() -> None:
    """Main training function with InternVL-specific logic."""
    args = parse_args()
    
    # Ensure weights are downloaded
    _ensure_weights(args.model_id, args.local_dir)
    
    # Load config from folder
    from omegaconf import OmegaConf
    config_path = Path(args.folder) / f"config_{args.dataset}.yml"
    if not config_path.exists():
        config_path = Path(args.folder) / "config_coco.yml"
    
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({})
    
    # Override config with CLI args for model loading
    if not hasattr(cfg, "HF"):
        cfg.HF = OmegaConf.create({})
    
    cfg.HF.MODEL_ID = args.model_id
    cfg.HF.PROCESSOR_ID = args.processor_id
    cfg.HF.LOCAL_DIR = args.local_dir
    cfg.HF.TRUST_REMOTE_CODE = True
    cfg.HF.ATTN_IMPLEMENTATION = args.attn_implementation
    cfg.HF.TORCH_DTYPE = "bfloat16"
    cfg.HF.LOW_CPU_MEM_USAGE = True
    
    # LoRA config
    if not hasattr(cfg.HF, "LORA"):
        cfg.HF.LORA = OmegaConf.create({})
    cfg.HF.LORA.R = args.lora_r
    cfg.HF.LORA.ALPHA = args.lora_alpha
    cfg.HF.LORA.DROPOUT = args.lora_dropout
    cfg.HF.LORA.TARGET_MODULES = args.lora_target_modules
    
    # Load model and processor using generic loader
    print(f"[InternVL] Loading model from {args.local_dir}")
    model, _ = _load_model_and_processor(cfg.HF)
    
    # Load tokenizer separately with InternVL settings
    tokenizer = AutoTokenizer.from_pretrained(
        args.local_dir,
        trust_remote_code=True,
        use_fast=False  # InternVL official example uses use_fast=False
    )
    
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
    
    # Create InternVL-specific collator
    system_prompt = cfg.get("SYSTEM_PROMPT", "")
    user_prompt = cfg.get("USER_PROMPT", "Describe the image in detail.")
    
    collator = InternVLCollator(
        tokenizer=tokenizer,
        model_path=args.local_dir,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        input_size=448,
        max_num_tiles=12,
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
        tokenizer=tokenizer,
    )
    
    # Train
    print("[InternVL] Starting training")
    trainer.train()
    
    # Best model is already saved by trainer's callback
    print(f"[InternVL] Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()

