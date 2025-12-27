import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


"""Qwen3-VL 训练脚本 - 使用 HuggingFace Trainer
    python tools/train_hf_trainer.py \
        --folder PureT/experiments/ByteCaption_XE_qwen \
        --dataset coco \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --processor_id Qwen/Qwen3-VL-8B-Instruct \
        --local_dir ./Qwen3-VL-8B-Instruct \
        --train_samples 0 \
        --val_samples 10 \
        --eval_steps 5 \
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

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.tokenization_utils_base import BatchEncoding

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

from transformers import Qwen3VLForConditionalGeneration  # type: ignore 


def parse_args():
    parser = argparse.ArgumentParser(description="HF caption training via Transformers Trainer")
    parser.add_argument("--folder", type=str, required=True, help="Experiment folder (contains config_*.yml)")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "flickr8k"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--best_metric", type=str, default="SPICE")
    parser.add_argument("--keep_full_metrics", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ByteCaption")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--processor_id", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default=None)
    # Qwen3-VL 仅使用 chat 模式
    parser.add_argument("--train_system_prompt", type=str, default=None)
    parser.add_argument("--train_user_prompt", type=str, default=None)
    parser.add_argument("--train_max_length", type=int, default=512)
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
    # Qwen3-VL 固定使用 chat 模式
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

    if prompt_source == "openrouter" and not user_prompt:
        user_prompt = (
            "You are given a possibly corrupted JPEG image. "
            "If you can decode it, output a short COCO-style caption. "
            f"If you cannot decode it, output exactly: {placeholder} "
            "Output only the caption with no extra text."
        )
    return prompt_source, system_prompt, user_prompt, placeholder


def _resolve_training_prompts(hf_cfg) -> Tuple[str, str]:
    if hf_cfg is None:
        return "", ""
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


def _build_model_kwargs(hf_cfg) -> dict:
    model_kwargs = {
        "trust_remote_code": bool(getattr(hf_cfg, "TRUST_REMOTE_CODE", False)),
        "use_safetensors": bool(getattr(hf_cfg, "SAFE_SERIALIZATION", True)),
    }
    torch_dtype = _resolve_torch_dtype(getattr(hf_cfg, "TORCH_DTYPE", ""))
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if bool(getattr(hf_cfg, "LOW_CPU_MEM_USAGE", False)):
        model_kwargs["low_cpu_mem_usage"] = True
    attn_impl = str(getattr(hf_cfg, "ATTN_IMPLEMENTATION", "") or "").strip()
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    return model_kwargs


def _strip_attn_implementation(model_kwargs: dict) -> dict:
    if "attn_implementation" not in model_kwargs:
        return model_kwargs
    updated = dict(model_kwargs)
    updated.pop("attn_implementation", None)
    return updated


def _should_retry_without_attn_implementation(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "attn_implementation" in msg
        or "flash_attention" in msg
        or "flashattention" in msg
        or "flash attention" in msg
    )


def _from_pretrained_with_attn_fallback(model_cls, load_from: str, model_kwargs: dict):
    try:
        return model_cls.from_pretrained(load_from, **model_kwargs)
    except Exception as exc:
        if _should_retry_without_attn_implementation(exc):
            print("[HF] attn_implementation unsupported; retrying without it.")
            return model_cls.from_pretrained(load_from, **_strip_attn_implementation(model_kwargs))
        raise

def _load_processor(load_from: str, trust_remote_code: bool):
    try:
        processor = AutoProcessor.from_pretrained(
            load_from, 
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=True
        )
        return processor
    except (TypeError, AttributeError) as exc:
        # Handle missing image tokens for InternVL
        if "start_image_token" in str(exc) or "Qwen2TokenizerFast" in str(exc):
            from transformers import AutoTokenizer, AutoImageProcessor
            tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=trust_remote_code)
            image_processor = AutoImageProcessor.from_pretrained(load_from, trust_remote_code=trust_remote_code)
            
            # Add missing image token attributes for InternVL
            if not hasattr(tokenizer, 'start_image_token'):
                tokenizer.start_image_token = "<img>"
                tokenizer.end_image_token = "</img>"
                tokenizer.context_image_token = "<IMG_CONTEXT>"
                tokenizer.video_token = "<video>"
                # Get or create token IDs
                if "<img>" not in tokenizer.get_vocab():
                    tokenizer.add_tokens(["<img>", "</img>", "<IMG_CONTEXT>", "<video>"])
                tokenizer.start_image_token_id = tokenizer.convert_tokens_to_ids("<img>")
                tokenizer.end_image_token_id = tokenizer.convert_tokens_to_ids("</img>")
                tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            
            return AutoProcessor.from_pretrained(
                load_from,
                tokenizer=tokenizer,
                image_processor=image_processor,
                trust_remote_code=trust_remote_code
            )
        # Fallback without fix_mistral_regex if that was the issue
        return AutoProcessor.from_pretrained(load_from, trust_remote_code=trust_remote_code)

def _load_model_and_processor(hf_cfg):
    def _load_with_safetensor_retry(model_cls, allow_retry: bool = True):
        try:
            return _from_pretrained_with_attn_fallback(model_cls, load_from, model_kwargs)
        except OSError:
            if allow_retry and model_kwargs.get("use_safetensors", True):
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["use_safetensors"] = False
                return _from_pretrained_with_attn_fallback(model_cls, load_from, fallback_kwargs)
            raise

    model_id = getattr(hf_cfg, "MODEL_ID", "")
    processor_id = getattr(hf_cfg, "PROCESSOR_ID", "") or model_id
    local_dir = getattr(hf_cfg, "LOCAL_DIR", None)
    load_from = local_dir if (local_dir and os.path.isdir(local_dir)) else model_id
    processor_load_from = local_dir if (local_dir and os.path.isdir(local_dir)) else processor_id
    trust_remote_code = bool(getattr(hf_cfg, "TRUST_REMOTE_CODE", False))
    processor = _load_processor(
        processor_load_from, trust_remote_code=trust_remote_code
    )

    model_kwargs = _build_model_kwargs(hf_cfg)
    use_safetensors = bool(getattr(hf_cfg, "SAFE_SERIALIZATION", True))

    # 仅支持 Qwen3-VL 模型
    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("Qwen3VLForConditionalGeneration not available. Please install the required transformers version.")
    
    model = _load_with_safetensor_retry(Qwen3VLForConditionalGeneration)

    # Qwen3-VL 使用左侧填充
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    if not use_safetensors and hasattr(model, "config"):
        model.config.use_safetensors = False

    if bool(getattr(hf_cfg, "GRADIENT_CHECKPOINTING", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    return model, processor


def _apply_lora(model, hf_cfg):
    lora_cfg = getattr(hf_cfg, "LORA", None)
    if lora_cfg is None:
        return model, False
    if not bool(getattr(lora_cfg, "ENABLED", False)):
        return model, False
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        print(f"[HF] LoRA disabled: peft unavailable ({exc})")
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
        print(f"[HF] LoRA target modules not found; disabling LoRA. ({exc})")
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


class HFTrainerCollator:
    """专用于 Qwen3-VL 的数据整理器"""
    def __init__(
        self,
        processor,
        use_chat_template: bool = False,
        system_prompt: str = "",
        user_prompt: str = "",
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        label_ignore: int = -100,
        seq_per_img: int = 1,
    ) -> None:
        self.processor = processor
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_length = int(max_length) if max_length else None
        self.truncation = bool(truncation) if truncation is not None else False
        self.label_ignore = int(label_ignore)
        self.seq_per_img = max(int(seq_per_img), 1)
        self._pad_token_id = None
        self._warned_truncation = False

    # Qwen3-VL 固定使用 chat 模式

    def _build_chat_text(self, caption: str, with_answer: bool) -> str:
        prompt = ""
        if self.system_prompt:
            prompt = self.system_prompt.strip()
        if self.user_prompt:
            if prompt:
                prompt = f"{prompt}\n{self.user_prompt.strip()}"
            else:
                prompt = self.user_prompt.strip()
        
        if with_answer:
            return f"{prompt}\n{caption}" if prompt else caption
        return prompt

    def _build_chat_messages(self, image: Any, caption: Optional[str]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        
        user_content: List[Dict[str, Any]] = []
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        
        if caption is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
        return messages

    def _resolve_truncation(self) -> Tuple[bool, Optional[int]]:
        truncation = self.truncation
        max_length = self.max_length
        return truncation, max_length

    def _mask_prompt_labels(
        self,
        labels: Optional[torch.Tensor],
        prompt_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if labels is None or prompt_mask is None or attention_mask is None:
            return
        padding_side = "right"
        if hasattr(self.processor, "tokenizer"):
            padding_side = getattr(self.processor.tokenizer, "padding_side", "right")
        elif hasattr(self.processor, "padding_side"):
            padding_side = getattr(self.processor, "padding_side", "right")
        seq_len = labels.size(1)
        full_lens = attention_mask.sum(dim=1).tolist()
        prompt_lens = prompt_mask.sum(dim=1).tolist()
        for i, plen in enumerate(prompt_lens):
            plen = int(plen)
            if plen <= 0:
                continue
            flen = int(full_lens[i]) if i < len(full_lens) else 0
            if flen <= 0:
                continue
            pad_len = seq_len - flen
            start = pad_len if padding_side == "left" else 0
            end = min(seq_len, start + plen)
            labels[i, start:end] = self.label_ignore

    def _safe_processor_call(self, **kwargs):
        try:
            return self.processor(**kwargs)
        except ValueError as exc:
            msg = str(exc).lower()
            if "image token count" in msg or "mismatch in `image` token count" in msg:
                if not self._warned_truncation:
                    print("[HF] Truncation caused image token mismatch; retrying without truncation.")
                    self._warned_truncation = True
                kwargs["truncation"] = False
                kwargs.pop("max_length", None)
                return self.processor(**kwargs)
            raise

    def _safe_apply_chat_template(self, conversation, **kwargs):
        try:
            return self.processor.apply_chat_template(conversation, **kwargs)
        except ValueError as exc:
            msg = str(exc).lower()
            if "image token count" in msg or "mismatch in `image` token count" in msg:
                if not self._warned_truncation:
                    print("[HF] Truncation caused image token mismatch; retrying without truncation.")
                    self._warned_truncation = True
                kwargs["truncation"] = False
                kwargs.pop("max_length", None)
                return self.processor.apply_chat_template(conversation, **kwargs)
            raise

    def __call__(self, batch: Sequence[Tuple[Any, ...]]):
        if self._pad_token_id is None:
            self._pad_token_id = getattr(self.processor, "pad_token_id", None)
            if self._pad_token_id is None and hasattr(self.processor, "tokenizer"):
                self._pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        indices, captions_list, _gv_feat, images = zip(*batch)
        expanded_images: List[Any] = []
        expanded_captions: List[str] = []
        for img, caps in zip(images, captions_list):
            selected = _select_captions(list(caps), self.seq_per_img)
            for cap in selected:
                expanded_images.append(img)
                expanded_captions.append(cap)

        truncation, max_length = self._resolve_truncation()

        # Qwen3-VL 使用 chat 模板
        if self.use_chat_template and hasattr(self.processor, "apply_chat_template"):
            full_messages = [
                self._build_chat_messages(img, cap)
                for img, cap in zip(expanded_images, expanded_captions)
            ]
            prompt_messages = [self._build_chat_messages(img, None) for img in expanded_images]

            full_inputs_kwargs = {
                "tokenize": True,
                "add_generation_prompt": False,
                "return_tensors": "pt",
                "return_dict": True,
                "padding": True,
                "truncation": truncation,
            }
            if truncation and max_length is not None:
                full_inputs_kwargs["max_length"] = max_length
            full_inputs = self._safe_apply_chat_template(full_messages, **full_inputs_kwargs)
            input_ids = full_inputs.get("input_ids")
            attention_mask = full_inputs.get("attention_mask")
            labels = input_ids.clone() if input_ids is not None else None
            if labels is not None and self._pad_token_id is not None:
                labels[labels == self._pad_token_id] = self.label_ignore

            prompt_inputs_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
                "padding": True,
                "truncation": truncation,
            }
            if truncation and max_length is not None:
                prompt_inputs_kwargs["max_length"] = max_length
            prompt_inputs = self._safe_apply_chat_template(prompt_messages, **prompt_inputs_kwargs)
            prompt_mask = prompt_inputs.get("attention_mask")
            self._mask_prompt_labels(labels, prompt_mask, attention_mask)

            inputs = full_inputs
        else:
            # 退后逻辑：使用简单文本+图像处理
            prompt_texts = [
                self._build_chat_text(cap, with_answer=False) for cap in expanded_captions
            ]
            full_texts = [
                self._build_chat_text(cap, with_answer=True) for cap in expanded_captions
            ]

            full_inputs_kwargs = {
                "text": full_texts,
                "images": expanded_images,
                "return_tensors": "pt",
                "padding": True,
                "truncation": truncation,
            }
            if truncation and max_length is not None:
                full_inputs_kwargs["max_length"] = max_length
            full_inputs = self._safe_processor_call(**full_inputs_kwargs)
            input_ids = full_inputs.get("input_ids")
            attention_mask = full_inputs.get("attention_mask")
            labels = input_ids.clone() if input_ids is not None else None
            if labels is not None and self._pad_token_id is not None:
                labels[labels == self._pad_token_id] = self.label_ignore

            prompt_inputs_kwargs = {
                "text": prompt_texts,
                "images": expanded_images,
                "return_tensors": "pt",
                "padding": True,
                "truncation": truncation,
            }
            if truncation and max_length is not None:
                prompt_inputs_kwargs["max_length"] = max_length
            prompt_inputs = self._safe_processor_call(**prompt_inputs_kwargs)
            prompt_mask = prompt_inputs.get("attention_mask")
            self._mask_prompt_labels(labels, prompt_mask, attention_mask)

            inputs = full_inputs

        batch_inputs: Dict[str, torch.Tensor] = {
            key: value for key, value in inputs.items() if key != "input_ids" and key != "attention_mask"
        }
        batch_inputs["input_ids"] = input_ids
        batch_inputs["attention_mask"] = attention_mask
        if labels is not None:
            batch_inputs["labels"] = labels
        return batch_inputs


class HFDecodeWrapper:
    def __init__(
        self,
        model,
        processor,
        generation_kwargs: Optional[dict] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        placeholder: str = "this is a dummy caption for an undecodable image",
        use_chat_template: bool = False,
    ) -> None:
        self.model = getattr(model, "module", model)
        self.processor = processor
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.placeholder = placeholder
        self.use_chat_template = use_chat_template

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _build_chat_messages(self, image) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        user_content: List[Dict[str, Any]] = []
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_text_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.user_prompt:
            messages.append({"role": "user", "content": self.user_prompt})
        elif self.system_prompt:
            messages.append({"role": "user", "content": ""})
        return messages

    def _compose_prompt_text(self) -> str:
        if self.system_prompt and self.user_prompt:
            return f"{self.system_prompt}\n{self.user_prompt}"
        return self.user_prompt or self.system_prompt or ""

    def _prepare_model_inputs(self, images: List[Any]) -> dict:
        if self.use_chat_template and hasattr(self.processor, "apply_chat_template"):
            messages = [self._build_chat_messages(image) for image in images]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        else:
            prompt_text = self._compose_prompt_text()
            if prompt_text:
                texts = [prompt_text for _ in images]
                inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            else:
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
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
        if getattr(self.model.config, "is_encoder_decoder", False):
            return generated_ids
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
        generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

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
        use_chat_template: bool = False,
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
        self.use_chat_template = use_chat_template
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
            print(f"[HF] Best adapter saved to {best_dir} ({metric_key}={value:.4f})")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if self.caption_evaler is None:
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if self.args.eval_strategy == "epoch" and self.state.epoch is not None:
            current_epoch = int(self.state.epoch) + 1
            if current_epoch % self.eval_interval_epochs != 0:
                return {}

        wrapper = HFDecodeWrapper(
            self.model,
            self.processor,
            generation_kwargs=self.generation_kwargs,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            placeholder=self.placeholder,
            use_chat_template=self.use_chat_template,
        )
        metrics = self.caption_evaler(wrapper, self.eval_name)
        prefixed = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(prefixed)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, prefixed)
        self._maybe_save_best_adapter(prefixed, metric_key_prefix)
        return prefixed


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
    model, lora_enabled = _apply_lora(model, hf_cfg)

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

    system_prompt, user_prompt = _resolve_training_prompts(hf_cfg)
    prompt_source, infer_system, infer_user, placeholder = _resolve_prompt_settings(hf_cfg)

    train_max_length = int(args.train_max_length) if args.train_max_length is not None else None
    train_truncation = bool(args.train_truncation) if args.train_truncation is not None else None

    collator = HFTrainerCollator(
        processor=processor,
        use_chat_template=bool(getattr(hf_cfg, "USE_CHAT_TEMPLATE", False)),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_length=train_max_length,
        truncation=train_truncation,
        label_ignore=int(getattr(hf_cfg, "TRAIN_LABEL_IGNORE", -100)),
        seq_per_img=int(getattr(cfg.DATA_LOADER, "SEQ_PER_IMG", 1)),
    )

    output_dir = args.output_dir or str(folder / "snapshot" / "hf_trainer")
    os.makedirs(output_dir, exist_ok=True)

    def _resolve_train_value(arg_value, _cfg_value):
        return arg_value

    label_smoothing = 0.0
    if str(getattr(cfg.LOSSES, "XE_TYPE", "")).lower() == "labelsmoothing":
        label_smoothing = float(getattr(cfg.LOSSES, "LABELSMOOTHING", 0.0))

    eval_steps = args.eval_steps if args.eval_steps and args.eval_steps > 0 else None
    save_steps = args.save_steps if args.save_steps and args.save_steps > 0 else eval_steps

    metric_for_best = f"eval_{args.best_metric}" if args.best_metric else None
    greater_is_better = True
    if args.best_metric and "loss" in args.best_metric.lower():
        greater_is_better = False

    torch_dtype = _resolve_torch_dtype(getattr(hf_cfg, "TORCH_DTYPE", ""))
    use_fp16 = torch_dtype == torch.float16
    use_bf16 = torch_dtype == torch.bfloat16

    eval_strategy = "no" if val_samples in (None, 0) else ("steps" if eval_steps else "epoch")
    if args.early_stop_patience and args.early_stop_patience > 0 and eval_strategy == "no":
        print("[HF] Early stopping disabled because eval is off (val_samples=0).")
        args.early_stop_patience = 0
    save_strategy = "no" if bool(getattr(hf_cfg.LORA, "ENABLED", False)) and not bool(getattr(hf_cfg.LORA, "SAVE_FULL_MODEL", True)) else eval_strategy

    training_kwargs = {
        "output_dir": output_dir,
        "max_steps": int(args.max_steps) if args.max_steps else -1,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "save_total_limit": int(getattr(cfg.SOLVER, "MAX_CHECKPOINTS", 2)),
        "remove_unused_columns": False,
        "report_to": [] if args.disable_wandb else ["wandb"],
        "run_name": args.wandb_name or folder.name,
        "load_best_model_at_end": bool(metric_for_best and save_strategy != "no"),
        "metric_for_best_model": metric_for_best,
        "greater_is_better": greater_is_better,
        "label_smoothing_factor": label_smoothing,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "dataloader_num_workers": int(getattr(cfg.DATA_LOADER, "NUM_WORKERS", 0)),
        "dataloader_pin_memory": bool(getattr(cfg.DATA_LOADER, "PIN_MEMORY", False)),
    }
    
    # Only set prefetch_factor and persistent_workers if num_workers > 0
    num_workers = int(getattr(cfg.DATA_LOADER, "NUM_WORKERS", 0))
    if num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = int(getattr(cfg.DATA_LOADER, "PREFETCH_FACTOR", 2))
        training_kwargs["dataloader_persistent_workers"] = bool(getattr(cfg.DATA_LOADER, "PERSISTENT_WORKERS", True))

    if args.grad_accum_steps is not None:
        training_kwargs["gradient_accumulation_steps"] = int(args.grad_accum_steps)
    if eval_steps is not None:
        training_kwargs["eval_steps"] = eval_steps
    if save_steps is not None:
        training_kwargs["save_steps"] = save_steps
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

    training_kwargs["lr_scheduler_type"] = "linear"
    training_kwargs["warmup_steps"] = 0
    training_kwargs["optim"] = "adamw_torch"
    training_kwargs["adam_beta1"] = float(getattr(cfg.SOLVER.ADAM, "BETAS", [0.9, 0.999])[0])
    training_kwargs["adam_beta2"] = float(getattr(cfg.SOLVER.ADAM, "BETAS", [0.9, 0.999])[1])
    training_kwargs["adam_epsilon"] = float(getattr(cfg.SOLVER.ADAM, "EPS", 1e-8))
    training_kwargs["max_grad_norm"] = float(getattr(cfg.SOLVER, "GRAD_CLIP", 0.0))
    training_kwargs["weight_decay"] = float(getattr(cfg.SOLVER, "WEIGHT_DECAY", 0.0))

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
        use_chat_template=bool(getattr(hf_cfg, "USE_CHAT_TEMPLATE", False)),
        eval_interval_epochs=int(cfg.SOLVER.TEST_INTERVAL) if eval_steps is None else 1,
        save_adapter_only=bool(getattr(hf_cfg.LORA, "ENABLED", False)) and not bool(getattr(hf_cfg.LORA, "SAVE_FULL_MODEL", True)),
        best_metric=args.best_metric,
    )

    callbacks = []
    if args.early_stop_patience and args.early_stop_patience > 0:
        try:
            from transformers import EarlyStoppingCallback

            callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(args.early_stop_patience)))
        except Exception as exc:
            print(f"[HF] EarlyStoppingCallback unavailable: {exc}")

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
