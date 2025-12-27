#!/usr/bin/env python3
"""InternVL3_5-8B LoRA training via pure Transformers Trainer.

This script reuses the ByteCaption datasets (no JSON export) and follows
the official InternVL chat-template usage for prompt construction.

Example:
  python tools/train_internvl_transformers_lora.py \
    --folder PureT/experiments/ByteCaption_XE_internvl \
    --dataset coco \
    --model_id InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
    --processor_id InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
    --local_dir InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
    --train_samples 0 \
    --val_samples 200 \
    --save_steps 200 \
    --save_total_limit 1 \
    --best_metric SPICE \
    --base_lr 1e-4 \
    --early_stop_patience 4 \
    --max_epoch 1 \
    --batch_size 1 \
    --test_batch_size 1 \
    --grad_accum_steps 8 \
    --gradient_checkpointing \
    --num_workers 8 \
    --train_max_length 3600 \
    --train_truncation 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --attn_implementation sdpa 
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None

try:
    from transformers import InternVLForConditionalGeneration
except Exception:  # pragma: no cover
    InternVLForConditionalGeneration = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "PureT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

from lib.config import cfg, cfg_from_file  # noqa: E402
from PureT.datasets_.coco_dataset import CocoDataset  # noqa: E402
from PureT.datasets_.flickr8k_dataset import Flickr8kDataset  # noqa: E402
from PureT.evaluation.evaler_coco import CocoEvaler  # noqa: E402
from PureT.evaluation.evaler_flickr8k import Flickr8kEvaler  # noqa: E402


IMAGE_PLACEHOLDER = "<image_placeholder>"


def parse_args():
    parser = argparse.ArgumentParser(description="InternVL LoRA training via Transformers Trainer")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "flickr8k"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--best_metric", type=str, default="SPICE")
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--processor_id", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--train_system_prompt", type=str, default=None)
    parser.add_argument("--train_user_prompt", type=str, default=None)
    parser.add_argument("--train_max_length", type=int, default=4096)
    parser.add_argument("--train_truncation", type=int, default=0, choices=[0, 1])

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", type=str, default=None)
    parser.add_argument("--lora_task_type", type=str, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    parser.add_argument("--lora_modules_to_save", nargs="+", default=None)
    parser.add_argument("--lora_save_full_model", action="store_true")

    parser.add_argument("--seq_per_img", type=int, default=None)
    parser.add_argument("--shuffle", type=int, default=None, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", type=int, default=None, choices=[0, 1])
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--test_interval", type=int, default=None)
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=1)
    return parser.parse_args()


def _load_config(folder: Path, dataset: str):
    config_file = "config_coco.yml" if dataset == "coco" else "config_flickr8k.yml"
    config_path = folder / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg_from_file(str(config_path))
    cfg.ROOT_DIR = str(folder)
    return config_path


def _apply_hf_overrides(args):
    hf_cfg = cfg.MODEL.HF
    hf_cfg.TRAINABLE = True
    hf_cfg.LORA.ENABLED = True

    if args.model_id:
        hf_cfg.MODEL_ID = args.model_id
    if args.processor_id:
        hf_cfg.PROCESSOR_ID = args.processor_id
    if args.local_dir:
        hf_cfg.LOCAL_DIR = args.local_dir
    if args.train_system_prompt is not None:
        hf_cfg.TRAIN_SYSTEM_PROMPT = args.train_system_prompt
    if args.train_user_prompt is not None:
        hf_cfg.TRAIN_USER_PROMPT = args.train_user_prompt
    if args.train_max_length is not None:
        hf_cfg.TRAIN_MAX_LENGTH = int(args.train_max_length)
    if args.train_truncation is not None:
        hf_cfg.TRAIN_TRUNCATION = bool(args.train_truncation)

    if args.lora_r is not None:
        hf_cfg.LORA.R = int(args.lora_r)
    if args.lora_alpha is not None:
        hf_cfg.LORA.ALPHA = int(args.lora_alpha)
    if args.lora_dropout is not None:
        hf_cfg.LORA.DROPOUT = float(args.lora_dropout)
    if args.lora_bias is not None:
        hf_cfg.LORA.BIAS = args.lora_bias
    if args.lora_task_type is not None:
        hf_cfg.LORA.TASK_TYPE = args.lora_task_type
    if args.lora_target_modules is not None:
        hf_cfg.LORA.TARGET_MODULES = args.lora_target_modules
    if args.lora_modules_to_save is not None:
        hf_cfg.LORA.MODULES_TO_SAVE = args.lora_modules_to_save
    if args.lora_save_full_model:
        hf_cfg.LORA.SAVE_FULL_MODEL = True
    if args.gradient_checkpointing:
        hf_cfg.GRADIENT_CHECKPOINTING = True
    if args.attn_implementation is not None:
        hf_cfg.ATTN_IMPLEMENTATION = args.attn_implementation


def _apply_dataloader_overrides(args):
    if args.seq_per_img is not None:
        cfg.DATA_LOADER.SEQ_PER_IMG = int(args.seq_per_img)
    else:
        cfg.DATA_LOADER.SEQ_PER_IMG = 1
    if args.shuffle is not None:
        cfg.DATA_LOADER.SHUFFLE = bool(args.shuffle)
    else:
        cfg.DATA_LOADER.SHUFFLE = True
    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_WORKERS = int(args.num_workers)
    if args.pin_memory is not None:
        cfg.DATA_LOADER.PIN_MEMORY = bool(args.pin_memory)
    if args.prefetch_factor is not None:
        cfg.DATA_LOADER.PREFETCH_FACTOR = int(args.prefetch_factor)


def _apply_solver_overrides(args):
    if args.max_epoch is not None:
        cfg.SOLVER.MAX_EPOCH = int(args.max_epoch)
    if args.test_interval is not None:
        cfg.SOLVER.TEST_INTERVAL = int(args.test_interval)
    if args.base_lr is not None:
        cfg.SOLVER.BASE_LR = float(args.base_lr)
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = int(args.batch_size)
    if args.test_batch_size is not None:
        cfg.TEST.BATCH_SIZE = int(args.test_batch_size)
    if args.grad_accum_steps is not None:
        args.grad_accum_steps = int(args.grad_accum_steps)


def _resolve_torch_dtype(dtype_value):
    if not dtype_value:
        return None
    if isinstance(dtype_value, str):
        lowered = dtype_value.strip().lower()
        if not lowered:
            return None
        if lowered == "auto":
            return "auto"
        if lowered in ("float16", "fp16", "half"):
            return torch.float16
        if lowered in ("bfloat16", "bf16"):
            return torch.bfloat16
        if lowered in ("float32", "fp32"):
            return torch.float32
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    return None


def _resolve_attn_impl(attn_impl: Optional[str]) -> Optional[str]:
    if not attn_impl:
        return None
    lowered = str(attn_impl).lower()
    if "flash" in lowered:
        try:
            import flash_attn  # noqa: F401
        except Exception:
            return "sdpa"
    return attn_impl


@contextmanager
def _hf_env(mirror: Optional[str], disable_proxy: bool):
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


def _load_processor_with_fallback(processor_id: str, trust_remote_code: bool):
    try:
        return AutoProcessor.from_pretrained(processor_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        msg = str(exc)
        if "start_image_token" not in msg and "start_image_token" not in repr(exc):
            raise
        try:
            return AutoProcessor.from_pretrained(
                processor_id, trust_remote_code=trust_remote_code, use_fast=False
            )
        except TypeError:
            return AutoProcessor.from_pretrained(processor_id, trust_remote_code=trust_remote_code)


def _load_model_and_processor(hf_cfg):
    model_id = getattr(hf_cfg, "MODEL_ID", "")
    processor_id = getattr(hf_cfg, "PROCESSOR_ID", "") or model_id
    local_dir = getattr(hf_cfg, "LOCAL_DIR", None)
    load_from = local_dir if (local_dir and os.path.isdir(local_dir)) else model_id
    processor_load_from = local_dir if (local_dir and os.path.isdir(local_dir)) else processor_id
    trust_remote_code = bool(getattr(hf_cfg, "TRUST_REMOTE_CODE", False))
    mirror = getattr(hf_cfg, "MIRROR", None)
    disable_proxy = bool(getattr(hf_cfg, "DISABLE_PROXY", False))

    torch_dtype = _resolve_torch_dtype(getattr(hf_cfg, "TORCH_DTYPE", "") if hf_cfg else "")
    if torch_dtype is None and torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": bool(getattr(hf_cfg, "LOW_CPU_MEM_USAGE", False)),
    }
    attn_impl = _resolve_attn_impl(getattr(hf_cfg, "ATTN_IMPLEMENTATION", None))
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    use_safetensors = bool(getattr(hf_cfg, "SAFE_SERIALIZATION", True))
    model_kwargs["use_safetensors"] = use_safetensors

    with _hf_env(mirror, disable_proxy):
        processor = _load_processor_with_fallback(processor_load_from, trust_remote_code=trust_remote_code)

        model = None
        if AutoModelForImageTextToText is not None:
            try:
                model = AutoModelForImageTextToText.from_pretrained(load_from, **model_kwargs)
            except TypeError:
                trimmed = dict(model_kwargs)
                trimmed.pop("attn_implementation", None)
                model = AutoModelForImageTextToText.from_pretrained(load_from, **trimmed)
        if model is None and InternVLForConditionalGeneration is not None:
            try:
                model = InternVLForConditionalGeneration.from_pretrained(load_from, **model_kwargs)
            except TypeError:
                trimmed = dict(model_kwargs)
                trimmed.pop("attn_implementation", None)
                model = InternVLForConditionalGeneration.from_pretrained(load_from, **trimmed)
        if model is None:
            raise RuntimeError(f"Failed to load InternVL model from {load_from}")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    if bool(getattr(hf_cfg, "GRADIENT_CHECKPOINTING", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    return model, processor


def _apply_lora(model, hf_cfg):
    lora_cfg = getattr(hf_cfg, "LORA", None)
    if lora_cfg is None or not bool(getattr(lora_cfg, "ENABLED", False)):
        return model, False
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        print(f"[InternVL] LoRA disabled: peft unavailable ({exc})")
        return model, False

    task_name = str(getattr(lora_cfg, "TASK_TYPE", "CAUSAL_LM")).upper()
    task_type = getattr(TaskType, task_name, TaskType.CAUSAL_LM)
    target_modules = list(getattr(lora_cfg, "TARGET_MODULES", []) or [])
    modules_to_save = list(getattr(lora_cfg, "MODULES_TO_SAVE", []) or [])
    lora_config = LoraConfig(
        r=int(getattr(lora_cfg, "R", 8)),
        lora_alpha=int(getattr(lora_cfg, "ALPHA", 16)),
        lora_dropout=float(getattr(lora_cfg, "DROPOUT", 0.05)),
        bias=str(getattr(lora_cfg, "BIAS", "none")),
        task_type=task_type,
        target_modules=target_modules or None,
        modules_to_save=modules_to_save or None,
    )
    try:
        model = get_peft_model(model, lora_config)
    except ValueError as exc:
        print(f"[InternVL] LoRA target modules not found; disabling LoRA. ({exc})")
        return model, False
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, True


def _select_captions(captions: Sequence[str], seq_per_img: int) -> List[str]:
    if not captions:
        return ["."]
    if seq_per_img <= 1:
        return [captions[0]]
    if len(captions) >= seq_per_img:
        return list(captions[:seq_per_img])
    repeat_times = seq_per_img // len(captions)
    remainder = seq_per_img % len(captions)
    return list(captions) * repeat_times + list(captions)[:remainder]


def _resolve_training_prompts(hf_cfg) -> Tuple[str, str]:
    train_system = str(getattr(hf_cfg, "TRAIN_SYSTEM_PROMPT", "") or "").strip()
    train_user = str(getattr(hf_cfg, "TRAIN_USER_PROMPT", "") or "").strip()
    if train_system or train_user:
        return train_system, train_user

    prompt_source = str(getattr(hf_cfg, "PROMPT_SOURCE", "") or "").strip().lower()
    if prompt_source == "openrouter":
        or_cfg = getattr(cfg.MODEL, "OPENROUTER", None)
        system_prompt = (or_cfg.SYSTEM_PROMPT if or_cfg else "").strip()
        user_prompt = (or_cfg.USER_PROMPT if or_cfg else "").strip()
        return system_prompt, user_prompt

    system_prompt = str(getattr(hf_cfg, "SYSTEM_PROMPT", "") or "").strip()
    user_prompt = str(getattr(hf_cfg, "USER_PROMPT", "") or "").strip()
    return system_prompt, user_prompt


def _resolve_prompt_settings(hf_cfg) -> Tuple[str, str, str, str]:
    prompt_source = str(getattr(hf_cfg, "PROMPT_SOURCE", "") or "").strip().lower() if hf_cfg else ""
    if prompt_source == "openrouter":
        or_cfg = getattr(cfg.MODEL, "OPENROUTER", None)
        system_prompt = (or_cfg.SYSTEM_PROMPT if or_cfg else "").strip()
        user_prompt = (or_cfg.USER_PROMPT if or_cfg else "").strip()
        placeholder = (or_cfg.PLACEHOLDER if or_cfg else "").strip()
    else:
        system_prompt = (hf_cfg.SYSTEM_PROMPT if hf_cfg else "").strip()
        user_prompt = (hf_cfg.USER_PROMPT if hf_cfg else "").strip()
        placeholder = (hf_cfg.PLACEHOLDER if hf_cfg else "").strip()

    if not placeholder:
        placeholder = "this is a dummy caption for an undecodable image"
    return prompt_source, system_prompt, user_prompt, placeholder


def _build_generation_kwargs(hf_cfg) -> dict:
    gen_cfg = hf_cfg.GENERATION if hf_cfg and hasattr(hf_cfg, "GENERATION") else None
    max_new_tokens = None
    if gen_cfg and hasattr(gen_cfg, "MAX_NEW_TOKENS"):
        try:
            max_new_tokens = int(gen_cfg.MAX_NEW_TOKENS)
        except Exception:
            max_new_tokens = None
    max_length = int(gen_cfg.MAX_LENGTH) if gen_cfg and hasattr(gen_cfg, "MAX_LENGTH") else 50
    num_beams = int(gen_cfg.NUM_BEAMS) if gen_cfg and hasattr(gen_cfg, "NUM_BEAMS") else 3
    generation_kwargs = {"num_beams": num_beams}
    if max_new_tokens is not None and max_new_tokens > 0:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    else:
        generation_kwargs["max_length"] = max_length
    return generation_kwargs


class InternVLCollator:
    def __init__(
        self,
        processor,
        system_prompt: str,
        user_prompt: str,
        max_length: Optional[int],
        truncation: bool,
        label_ignore: int = -100,
        seq_per_img: int = 1,
    ) -> None:
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_length = max_length
        self.truncation = truncation
        self.label_ignore = int(label_ignore)
        self.seq_per_img = max(int(seq_per_img), 1)
        self._pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)

    def _build_messages(self, caption: Optional[str]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        user_content = [{"type": "image", "image": IMAGE_PLACEHOLDER}]
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        messages.append({"role": "user", "content": user_content})
        if caption is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
        return messages

    def _apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def __call__(self, batch: Sequence[Tuple[Any, ...]]):
        indices, captions_list, _gv_feat, images = zip(*batch)

        expanded_images: List[Any] = []
        expanded_captions: List[str] = []
        for img, caps in zip(images, captions_list):
            selected = _select_captions(list(caps), self.seq_per_img)
            for cap in selected:
                expanded_images.append(img)
                expanded_captions.append(cap)

        full_messages = [self._build_messages(cap) for cap in expanded_captions]
        prompt_messages = [self._build_messages(None) for _ in expanded_captions]

        full_prompts = [self._apply_chat_template(msgs, add_generation_prompt=False) for msgs in full_messages]
        prompt_prompts = [self._apply_chat_template(msgs, add_generation_prompt=True) for msgs in prompt_messages]

        full_inputs = self.processor(
            images=expanded_images,
            text=full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length if self.truncation else None,
        )
        prompt_inputs = self.processor(
            images=expanded_images,
            text=prompt_prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length if self.truncation else None,
        )

        input_ids = full_inputs.get("input_ids")
        attention_mask = full_inputs.get("attention_mask")
        labels = input_ids.clone() if input_ids is not None else None
        if labels is not None and self._pad_token_id is not None:
            labels[labels == self._pad_token_id] = self.label_ignore

        if labels is not None:
            for i in range(labels.shape[0]):
                if self._pad_token_id is None:
                    question_len = prompt_inputs["input_ids"][i].numel()
                else:
                    question_len = prompt_inputs["input_ids"][i].ne(self._pad_token_id).sum().item()
                labels[i, :question_len] = self.label_ignore

        batch_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_flags", "num_patches_list"):
            if key in full_inputs:
                batch_inputs[key] = full_inputs[key]
        batch_inputs.pop("token_type_ids", None)
        return batch_inputs


class InternVLDecodeWrapper:
    def __init__(
        self,
        model,
        processor,
        generation_kwargs: Optional[dict] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        placeholder: str = "this is a dummy caption for an undecodable image",
    ) -> None:
        self.model = getattr(model, "module", model)
        self.processor = processor
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.placeholder = placeholder

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _build_messages(self, image) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        user_content = [{"type": "image", "image": IMAGE_PLACEHOLDER}]
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _prepare_model_inputs(self, images: List[Any]) -> dict:
        messages = [self._build_messages(image) for image in images]
        prompts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages
        ]
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else None
        if torch.is_tensor(pixel_values) and pixel_values.is_floating_point():
            target_dtype = next(self.model.parameters()).dtype
            if pixel_values.dtype != target_dtype:
                inputs["pixel_values"] = pixel_values.to(dtype=target_dtype)
        return inputs

    def _trim_generated_ids(self, generated_ids: torch.Tensor, inputs: dict) -> torch.Tensor:
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if input_ids is None:
            return generated_ids
        prompt_len = input_ids.shape[1]
        if generated_ids.shape[1] <= prompt_len:
            return generated_ids
        return generated_ids[:, prompt_len:]

    def decode_beam(self, **kwargs):
        images = kwargs[cfg.PARAM.ATT_FEATS]
        beam_size = kwargs.get("BEAM_SIZE", self.generation_kwargs.get("num_beams", 3))

        valid_images_with_indices = [(i, img) for i, img in enumerate(images) if img is not None]
        if not valid_images_with_indices:
            return [self.placeholder for _ in range(len(images))], None
        original_indices, valid_images = zip(*valid_images_with_indices)

        inputs = self._prepare_model_inputs(list(valid_images))
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["num_beams"] = beam_size
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = self._trim_generated_ids(generated_ids, inputs)

        if hasattr(self.processor, "batch_decode"):
            generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            tokenizer = getattr(self.processor, "tokenizer", None)
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True) if tokenizer else []

        final_captions = [self.placeholder for _ in range(len(images))]
        for idx, caption in zip(original_indices, generated_captions):
            final_captions[idx] = caption.strip() if caption.strip() else self.placeholder
        return final_captions, None

    def decode(self, **kwargs):
        kwargs["BEAM_SIZE"] = 1
        return self.decode_beam(**kwargs)


class CaptionTrainer(Trainer):
    def __init__(
        self,
        *args,
        evaler=None,
        eval_name: str = "val",
        processor=None,
        generation_kwargs: Optional[dict] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        placeholder: str = "this is a dummy caption for an undecodable image",
        eval_interval_epochs: int = 1,
        save_adapter_only: bool = False,
        best_metric: Optional[str] = None,
        **kwargs,
    ):
        tokenizer = kwargs.get("tokenizer", None)
        try:
            super().__init__(*args, **kwargs)
        except TypeError as exc:
            if "tokenizer" not in str(exc):
                raise
            kwargs.pop("tokenizer", None)
            super().__init__(*args, **kwargs)
            self.tokenizer = tokenizer
        self.caption_evaler = evaler
        self.eval_name = eval_name
        self.processor = processor
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.placeholder = placeholder
        self.eval_interval_epochs = max(int(eval_interval_epochs), 1)
        self.save_adapter_only = save_adapter_only
        self.best_metric = best_metric
        self._best_metric_value = None

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        if self.save_adapter_only:
            os.makedirs(output_dir, exist_ok=True)
            try:
                self.model.save_pretrained(output_dir)
            except Exception:
                super().save_model(output_dir, _internal_call)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            return
        super().save_model(output_dir, _internal_call)

    def _maybe_save_best_adapter(self, metrics: Dict[str, Any], metric_key_prefix: str):
        if not self.save_adapter_only or not self.best_metric:
            return
        metric_key = f"{metric_key_prefix}_{self.best_metric}"
        if metric_key not in metrics:
            return
        value = metrics[metric_key]
        if value is None:
            return
        improved = self._best_metric_value is None or value > self._best_metric_value
        if improved:
            self._best_metric_value = value
            root_dir = cfg.ROOT_DIR or self.args.output_dir
            best_dir = os.path.join(root_dir, "snapshot", "best_lora")
            os.makedirs(best_dir, exist_ok=True)
            self.save_model(best_dir)
            print(f"[InternVL] Best adapter saved to {best_dir} ({metric_key}={value:.4f})")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if self.caption_evaler is None:
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if self.args.eval_strategy == "epoch" and self.state.epoch is not None:
            current_epoch = int(self.state.epoch) + 1
            if current_epoch % self.eval_interval_epochs != 0:
                return {}

        wrapper = InternVLDecodeWrapper(
            self.model,
            self.processor,
            generation_kwargs=self.generation_kwargs,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            placeholder=self.placeholder,
        )
        metrics = self.caption_evaler(wrapper, self.eval_name)
        prefixed = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(prefixed)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, prefixed)
        self._maybe_save_best_adapter(prefixed, metric_key_prefix)
        return prefixed


def _resolve_train_value(cli_value, cfg_value):
    return cli_value if cli_value is not None else cfg_value


def main():
    args = parse_args()
    folder = Path(args.folder)
    _load_config(folder, args.dataset)

    _apply_hf_overrides(args)
    _apply_dataloader_overrides(args)
    _apply_solver_overrides(args)

    set_seed(int(getattr(cfg, "SEED", 42)))

    hf_cfg = cfg.MODEL.HF
    if not str(getattr(hf_cfg, "TORCH_DTYPE", "") or "").strip():
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            hf_cfg.TORCH_DTYPE = "bfloat16"

    model, processor = _load_model_and_processor(hf_cfg)
    model, _lora_enabled = _apply_lora(model, hf_cfg)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_samples = args.train_samples if args.train_samples > 0 else None
    val_samples = args.val_samples if args.val_samples > 0 else None

    if args.dataset == "coco":
        train_set = CocoDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            max_samples=train_samples,
            return_captions=True,
            return_pil=True,
        )
        val_set = CocoDataset(
            image_ids_path=cfg.DATA_LOADER.VAL_ID,
            input_seq=None,
            target_seq=None,
            gv_feat_path=cfg.DATA_LOADER.VAL_GV_FEAT,
            seq_per_img=1,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            max_samples=val_samples,
            return_captions=True,
            return_pil=True,
        )
        evaler = CocoEvaler(cfg.DATA_LOADER.VAL_ID, cfg.DATA_LOADER.VAL_GV_FEAT, cfg.DATA_LOADER.VAL_ATT_FEATS, None, max_samples=val_samples)
    else:
        train_set = Flickr8kDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            max_samples=train_samples,
            return_captions=True,
            return_pil=True,
        )
        val_set = Flickr8kDataset(
            image_ids_path=cfg.DATA_LOADER.VAL_ID,
            input_seq=None,
            target_seq=None,
            gv_feat_path=cfg.DATA_LOADER.VAL_GV_FEAT,
            seq_per_img=1,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            max_samples=val_samples,
            return_captions=True,
            return_pil=True,
        )
        evaler = Flickr8kEvaler(cfg.DATA_LOADER.VAL_ID, cfg.DATA_LOADER.VAL_GV_FEAT, cfg.DATA_LOADER.VAL_ATT_FEATS, None, max_samples=val_samples)

    train_system, train_user = _resolve_training_prompts(hf_cfg)
    _prompt_source, infer_system, infer_user, placeholder = _resolve_prompt_settings(hf_cfg)

    train_max_length = int(args.train_max_length) if args.train_max_length is not None else None
    train_truncation = bool(args.train_truncation) if args.train_truncation is not None else False

    collator = InternVLCollator(
        processor=processor,
        system_prompt=train_system,
        user_prompt=train_user,
        max_length=train_max_length,
        truncation=train_truncation,
        seq_per_img=int(cfg.DATA_LOADER.SEQ_PER_IMG),
    )

    output_dir = args.output_dir or str(folder / "snapshot" / "hf_lora")
    os.makedirs(output_dir, exist_ok=True)

    # 确保 eval_strategy 和 save_strategy 匹配
    # 如果设置了 save_steps 但没有设置 eval_steps，自动同步
    if args.save_steps is not None and args.eval_steps is None:
        args.eval_steps = args.save_steps
        print(f"[Info] Auto-syncing eval_steps to save_steps: {args.save_steps}")
    
    eval_strategy = "steps" if args.eval_steps is not None else "epoch"
    save_strategy = "steps" if args.save_steps is not None else eval_strategy
    metric_for_best = f"eval_{args.best_metric}" if args.best_metric else None
    greater_is_better = True if args.best_metric else None
    save_total_limit = max(int(args.save_total_limit), 1)

    training_kwargs = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "remove_unused_columns": False,
        "report_to": [] if args.disable_wandb else ["wandb"],
        "load_best_model_at_end": bool(metric_for_best and save_strategy != "no"),
        "metric_for_best_model": metric_for_best,
        "greater_is_better": greater_is_better,
        "fp16": False,
        "bf16": bool(getattr(hf_cfg, "TORCH_DTYPE", "") == "bfloat16"),
        "dataloader_num_workers": int(getattr(cfg.DATA_LOADER, "NUM_WORKERS", 0)),
        "dataloader_pin_memory": bool(getattr(cfg.DATA_LOADER, "PIN_MEMORY", False)),
    }

    num_workers = int(getattr(cfg.DATA_LOADER, "NUM_WORKERS", 0))
    if num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = int(getattr(cfg.DATA_LOADER, "PREFETCH_FACTOR", 2))
        training_kwargs["dataloader_persistent_workers"] = bool(getattr(cfg.DATA_LOADER, "PERSISTENT_WORKERS", True))

    if args.grad_accum_steps is not None:
        training_kwargs["gradient_accumulation_steps"] = int(args.grad_accum_steps)
    if args.eval_steps is not None:
        training_kwargs["eval_steps"] = int(args.eval_steps)
    if args.save_steps is not None:
        training_kwargs["save_steps"] = int(args.save_steps)
    if args.log_steps is not None:
        training_kwargs["logging_steps"] = int(args.log_steps)
        training_kwargs["logging_first_step"] = True

    train_batch = _resolve_train_value(args.batch_size, cfg.TRAIN.BATCH_SIZE)
    if train_batch is not None:
        training_kwargs["per_device_train_batch_size"] = int(train_batch)
    eval_batch = _resolve_train_value(args.test_batch_size, cfg.TEST.BATCH_SIZE)
    if eval_batch is not None:
        training_kwargs["per_device_eval_batch_size"] = int(eval_batch)
    learning_rate = _resolve_train_value(args.base_lr, cfg.SOLVER.BASE_LR)
    if learning_rate is not None:
        training_kwargs["learning_rate"] = float(learning_rate)
    num_train_epochs = _resolve_train_value(args.max_epoch, cfg.SOLVER.MAX_EPOCH)
    if num_train_epochs is not None:
        training_kwargs["num_train_epochs"] = float(num_train_epochs)
    if args.max_steps is not None:
        training_kwargs["max_steps"] = int(args.max_steps)

    training_args = TrainingArguments(**training_kwargs)
    generation_kwargs = _build_generation_kwargs(hf_cfg)

    trainer = CaptionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        tokenizer=processor,
        evaler=evaler,
        eval_name="val",
        processor=processor,
        generation_kwargs=generation_kwargs,
        system_prompt=infer_system,
        user_prompt=infer_user,
        placeholder=placeholder,
        eval_interval_epochs=int(cfg.SOLVER.TEST_INTERVAL) if args.eval_steps is None else 1,
        save_adapter_only=bool(getattr(hf_cfg.LORA, "ENABLED", False)) and not bool(getattr(hf_cfg.LORA, "SAVE_FULL_MODEL", True)),
        best_metric=args.best_metric,
    )

    callbacks = []
    if args.early_stop_patience and args.early_stop_patience > 0:
        try:
            from transformers import EarlyStoppingCallback

            callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(args.early_stop_patience)))
        except Exception as exc:
            print(f"[InternVL] EarlyStoppingCallback unavailable: {exc}")
    if callbacks:
        for callback in callbacks:
            trainer.add_callback(callback)

    trainer.train(resume_from_checkpoint=args.resume_from)

    if bool(getattr(hf_cfg.LORA, "ENABLED", False)) and not bool(getattr(hf_cfg.LORA, "SAVE_FULL_MODEL", True)):
        root_dir = cfg.ROOT_DIR or output_dir
        final_dir = os.path.join(root_dir, "snapshot", "final_lora")
        os.makedirs(final_dir, exist_ok=True)
        trainer.save_model(final_dir)


if __name__ == "__main__":
    main()
