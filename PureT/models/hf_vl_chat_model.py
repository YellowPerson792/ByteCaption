from __future__ import annotations

import os
import io
import base64
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    InternVLForConditionalGeneration,
)
try:
    from transformers import (
        Glm4vForConditionalGeneration,
        Mistral3ForConditionalGeneration,
        MistralCommonBackend
    )
except ImportError:
    Glm4vForConditionalGeneration = None
    Mistral3ForConditionalGeneration = None
    MistralCommonBackend = None
    
from peft import PeftModel

from lib.config import cfg


def _get_device(cfg_device: Optional[str]) -> torch.device:
    if cfg_device and str(cfg_device).lower() != "auto":
        return torch.device(cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _ensure_image_token(text: str, image_token: str) -> str:
    if image_token and image_token in text:
        return text
    if text:
        return f"{text}\n{image_token}"
    return image_token


def _load_processor_with_fallback(processor_id: str, trust_remote_code: bool, model_id_lower: str = ""):
    # Special handling for Mistral/Ministral
    if "mistral" in model_id_lower or "ministral" in model_id_lower:
        try:
            return MistralCommonBackend.from_pretrained(processor_id, trust_remote_code=trust_remote_code)
        except Exception as exc:
            print(f"[WARNING] Failed to load MistralCommonBackend: {exc}. Falling back to AutoProcessor.")
    
    try:
        return AutoProcessor.from_pretrained(processor_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        # Retry with use_fast=False for tokenizer issues
        if "start_image_token" in str(exc):
            return AutoProcessor.from_pretrained(
                processor_id, trust_remote_code=trust_remote_code, use_fast=False
            )
        raise


class HFVLChatModel(nn.Module):
    """Generic HF vision-language captioning wrapper (Qwen/InternVL/GLM/etc.)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        processor_id: Optional[str] = None,
        local_dir: Optional[str] = None,
        generation_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        use_safetensors: Optional[bool] = None,
    ):
        super().__init__()
        hf_cfg = getattr(cfg.MODEL, "HF", None)
        self.model_id = model_id or (hf_cfg.MODEL_ID if hf_cfg else "")
        self.processor_id = processor_id or (hf_cfg.PROCESSOR_ID if hf_cfg else "") or self.model_id
        self.local_dir = local_dir or (hf_cfg.LOCAL_DIR if hf_cfg else None)
        self.trust_remote_code = trust_remote_code if trust_remote_code is not None else (
            hf_cfg.TRUST_REMOTE_CODE if hf_cfg else False
        )
        self.use_safetensors = use_safetensors if use_safetensors is not None else (
            hf_cfg.SAFE_SERIALIZATION if hf_cfg else True
        )
        self.torch_dtype = self._resolve_torch_dtype(getattr(hf_cfg, "TORCH_DTYPE", "") if hf_cfg else "")
        if self.torch_dtype is None and torch.cuda.is_available():
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.low_cpu_mem_usage = bool(getattr(hf_cfg, "LOW_CPU_MEM_USAGE", False)) if hf_cfg else False
        self.use_chat_template = bool(getattr(hf_cfg, "USE_CHAT_TEMPLATE", False)) if hf_cfg else False
        self.image_placeholder = str(getattr(hf_cfg, "IMAGE_PLACEHOLDER", "<image_placeholder>") or "<image_placeholder>")
        self.image_token = str(getattr(hf_cfg, "IMAGE_TOKEN", "<image>") or "<image>")
        self.prompt_source, self.system_prompt, self.user_prompt, self.placeholder = self._resolve_prompt_settings(hf_cfg)
        self.attn_implementation = str(getattr(hf_cfg, "ATTN_IMPLEMENTATION", "") or "") if hf_cfg else ""
        gen_cfg = hf_cfg.GENERATION if hf_cfg and hasattr(hf_cfg, "GENERATION") else None
        self.generation_kwargs = generation_kwargs or self._resolve_generation_kwargs(gen_cfg)

        cfg_device = hf_cfg.DEVICE if hf_cfg and hasattr(hf_cfg, "DEVICE") else None
        self.device = _get_device(device or cfg_device)

        mirror = getattr(hf_cfg, "MIRROR", None) if hf_cfg else None
        disable_proxy = bool(getattr(hf_cfg, "DISABLE_PROXY", False)) if hf_cfg else False

        with _hf_env(mirror, disable_proxy):
            processor_load_from = self._resolve_processor_source()
            model_id_lower = self.model_id.lower()
            self.processor = _load_processor_with_fallback(processor_load_from, trust_remote_code=self.trust_remote_code, model_id_lower=model_id_lower)
            self.model = self._load_model()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        self.model.to(self.device)
        self.model.eval()
        
        # Detect if this is a GLM model for special handling
        self.is_glm_model = self._detect_glm_model()
        self.is_ministral_model = self._detect_ministral_model()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HFVLCaptionModel is inference-only in this pipeline.")

    def load_lora_adapter(self, adapter_dir: str) -> bool:
        """Load PEFT LoRA adapter for the model during evaluation."""
        
        if not os.path.isdir(adapter_dir):
            return False
        try:
            # Wrap model with PEFT for loading adapter
            peft_model = PeftModel.from_pretrained(self.model, adapter_dir)
            # Merge the adapter into the model for inference
            self.model = peft_model.merge_and_unload()
            return True
        except Exception as exc:
            print(f"[GLM] Failed to load LoRA adapter: {exc}")
            return False

    def _truncate_to_first_sentence(self, text: str) -> str:
        """Keep only the first sentence."""
        import re
        if not text:
            return text
        # Split by sentence terminators and get first non-empty sentence
        for seg in re.split(r'[.!?]', text):
            seg = seg.strip()
            if seg:
                return seg
        return text

    def decode_beam(self, **kwargs):
        images = kwargs[cfg.PARAM.ATT_FEATS]
        beam_size = kwargs.get("BEAM_SIZE", self.generation_kwargs.get("num_beams", 3))

        original_indices, valid_images = self._prepare_inputs(images)
        dummy_caption = self.placeholder
        if not valid_images:
            return [dummy_caption for _ in range(len(images))], None

        try:
            inputs = self._prepare_model_inputs(valid_images)
        except Exception as e:
            # If model input preparation fails (e.g., due to extreme aspect ratio),
            # return dummy captions for all images
            print(f"[MINISTRAL] WARNING: _prepare_model_inputs failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return [dummy_caption for _ in range(len(images))], None

        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["num_beams"] = beam_size

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = self._trim_generated_ids(generated_ids, inputs)
        captions = self._decode_ids(generated_ids)

        final_captions = [dummy_caption for _ in range(len(images))]
        for idx, caption in zip(original_indices, captions):
            cleaned = caption.strip()
            # Only truncate to first sentence for GLM models
            if self.is_glm_model:
                cleaned = self._truncate_to_first_sentence(cleaned)
            final_captions[idx] = cleaned if cleaned else dummy_caption
        return final_captions, None

    def decode(self, **kwargs):
        kwargs["BEAM_SIZE"] = 1
        return self.decode_beam(**kwargs)

    def _resolve_processor_source(self) -> str:
        if not self.local_dir or not os.path.isdir(self.local_dir):
            return self.processor_id
        candidates = ("processor_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")
        if any(os.path.exists(os.path.join(self.local_dir, name)) for name in candidates):
            return self.local_dir
        return self.processor_id

    def _load_model(self):
        load_from = self.local_dir if (self.local_dir and os.path.isdir(self.local_dir)) else self.model_id
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        if self.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        attn_impl = self._resolve_attn_impl()
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        if self.use_safetensors is not None:
            model_kwargs["use_safetensors"] = bool(self.use_safetensors)

        model = None
        lowered = str(load_from).lower()
        # Try GLM with dedicated Glm4vForConditionalGeneration class first
        if "glm" in lowered and Glm4vForConditionalGeneration is not None:
            model = self._safe_from_pretrained(Glm4vForConditionalGeneration, load_from, model_kwargs)
        # Try Qwen with CausalLM
        if model is None and "qwen" in lowered:
            model = self._safe_from_pretrained(AutoModelForCausalLM, load_from, model_kwargs)
        # Then try InternVL
        if model is None and "internvl" in lowered and InternVLForConditionalGeneration is not None:
            model = self._safe_from_pretrained(InternVLForConditionalGeneration, load_from, model_kwargs)
        # Mistral/Ministral prefers ImageTextToText
        if model is None and ("mistral" in lowered or "ministral" in lowered):
            model = self._safe_from_pretrained(Mistral3ForConditionalGeneration, load_from, model_kwargs)
        # Then try Vision2Seq
        if model is None and AutoModelForVision2Seq is not None:
            model = self._safe_from_pretrained(AutoModelForVision2Seq, load_from, model_kwargs)
        # Fallback to CausalLM for unknown models
        if model is None:
            model = self._safe_from_pretrained(AutoModelForCausalLM, load_from, model_kwargs)
        # Last resort: AutoModel (may not have generate)
        if model is None:
            model = self._safe_from_pretrained(AutoModel, load_from, model_kwargs)
        if model is None:
            raise RuntimeError(f"Failed to load model from {load_from}")
        return model

    def _safe_from_pretrained(self, model_cls, load_from: str, model_kwargs: dict):
        try:
            return model_cls.from_pretrained(load_from, **model_kwargs)
        except TypeError as exc:
            if "attn_implementation" in str(exc):
                trimmed = dict(model_kwargs)
                trimmed.pop("attn_implementation", None)
                return model_cls.from_pretrained(load_from, **trimmed)
            return None
        except Exception:
            return None

    def _detect_glm_model(self) -> bool:
        """Check if loaded model is a GLM model."""
        model_id_lower = f"{self.model_id or ''} {self.local_dir or ''}".lower()
        return "glm" in model_id_lower

    def _detect_ministral_model(self) -> bool:
        lowered = f"{self.model_id or ''} {self.local_dir or ''}".lower()
        return "mistral" in lowered or "ministral" in lowered or isinstance(self.model, Mistral3ForConditionalGeneration)

    def _resolve_attn_impl(self) -> Optional[str]:
        attn_impl = (self.attn_implementation or "").strip()
        if not attn_impl:
            return None
        lowered = attn_impl.lower()
        if "flash" in lowered:
            try:
                import flash_attn  # noqa: F401
            except Exception:
                return "sdpa"
        return attn_impl

    def _prepare_inputs(self, images: Sequence) -> Tuple[List[int], List]:
        valid_images_with_indices = [(i, img) for i, img in enumerate(images) if img is not None]
        if not valid_images_with_indices:
            return [], []
        original_indices, valid_images = zip(*valid_images_with_indices)
        return list(original_indices), list(valid_images)

    def _compose_prompt_text(self) -> str:
        if self.system_prompt and self.user_prompt:
            return f"{self.system_prompt}\n{self.user_prompt}"
        return self.user_prompt or self.system_prompt or ""

    def _build_chat_prompt(self) -> Optional[str]:
        if not hasattr(self.processor, "apply_chat_template"):
            return None
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        user_content = [{"type": "image", "image": self.image_placeholder}]
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        messages.append({"role": "user", "content": user_content})
        try:
            return self._safe_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return None

    def _safe_apply_chat_template(self, conversation, **kwargs):
        """Apply chat template with model-specific parameters (e.g., GLM disable thinking)."""
        # For GLM models, disable thinking to avoid meta-thought outputs
        if self.is_glm_model:
            kwargs.setdefault("enable_thinking", False)
        try:
            return self.processor.apply_chat_template(conversation, **kwargs)
        except Exception as exc:
            # Retry without problematic parameters if needed
            if "enable_thinking" in str(exc):
                kwargs.pop("enable_thinking", None)
                return self.processor.apply_chat_template(conversation, **kwargs)
            raise

    def _prepare_model_inputs(self, images: List) -> dict:
        if self.is_ministral_model:
            return self._prepare_ministral_inputs(images)
        prompt = None
        use_chat_template = self._should_use_chat_template()
        if use_chat_template:
            prompt = self._build_chat_prompt()
        if prompt is None:
            prompt = _ensure_image_token(self._compose_prompt_text(), self.image_token)
        texts = [prompt for _ in images] if prompt else None

        try:
            if texts:
                inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
            else:
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
        except ValueError as exc:
            if use_chat_template or "placeholders" not in str(exc).lower():
                raise
            prompt = self._build_chat_prompt()
            if prompt is None:
                raise
            texts = [prompt for _ in images]
            inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        except TypeError:
            if texts:
                inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            else:
                inputs = self.processor(text=["" for _ in images], return_tensors="pt", padding=True)

        return inputs.to(self.device)

    def _image_to_data_url(self, image: any) -> str:
        if isinstance(image, torch.Tensor):
            # Inference path should pass PIL; tensor not supported here
            raise ValueError("Tensor images are not supported in chat-template path; pass PIL or path.")
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            with Image.open(image) as img_obj:
                img = img_obj.convert("RGB")
        else:
            raise ValueError("Image must be PIL.Image or path")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _prepare_ministral_inputs(self, images: List) -> dict:
        messages_batch = []
        for img in images:
            image_url = self._image_to_data_url(img)
            user_content = []
            if self.user_prompt:
                user_content.append({"type": "text", "text": self.user_prompt})
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

            conv: List[dict] = []
            if self.system_prompt:
                conv.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            conv.append({"role": "user", "content": user_content})
            messages_batch.append(conv)

        inputs = self._safe_apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=False,
        )

        pixel_values = inputs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor):
            inputs["pixel_values"] = pixel_values.to(dtype=torch.bfloat16)
            if inputs.get("image_sizes") is None:
                h, w = pixel_values.shape[-2:]
                inputs["image_sizes"] = [(h, w)] * pixel_values.shape[0]
        return inputs.to(self.device)

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

    def _decode_ids(self, token_ids: torch.Tensor) -> List[str]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
            return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(token_ids, skip_special_tokens=True)
        return ["" for _ in range(token_ids.shape[0])]

    def _resolve_generation_kwargs(self, gen_cfg) -> dict:
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
        if gen_cfg and hasattr(gen_cfg, "TEMPERATURE"):
            generation_kwargs["temperature"] = float(gen_cfg.TEMPERATURE)
        if gen_cfg and hasattr(gen_cfg, "TOP_P"):
            generation_kwargs["top_p"] = float(gen_cfg.TOP_P)
        if gen_cfg and hasattr(gen_cfg, "TOP_K"):
            generation_kwargs["top_k"] = int(gen_cfg.TOP_K)
        if gen_cfg and hasattr(gen_cfg, "REPETITION_PENALTY"):
            generation_kwargs["repetition_penalty"] = float(gen_cfg.REPETITION_PENALTY)
        return generation_kwargs

    def _resolve_prompt_settings(self, hf_cfg) -> Tuple[str, str, str, str]:
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

    def _should_use_chat_template(self) -> bool:
        if self.use_chat_template:
            return True
        lowered = f"{self.model_id} {self.local_dir or ''}".lower()
        return "internvl" in lowered

    def _resolve_torch_dtype(self, dtype_value):
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
