import os
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.config import cfg


def _get_device(cfg_device: Optional[str]) -> torch.device:
    if cfg_device and cfg_device.lower() != "auto":
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


class HFTextCaptionModel(nn.Module):
    """
    Text-only HF caption wrapper (ignores images, uses prompt-only generation).
    Accepts a list of images but does not condition on them.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        local_dir: Optional[str] = None,
        generation_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        use_safetensors: Optional[bool] = None,
    ):
        super().__init__()
        hf_cfg = getattr(cfg.MODEL, "HF", None)
        self.model_id = model_id or (hf_cfg.MODEL_ID if hf_cfg else "")
        self.tokenizer_id = tokenizer_id or (hf_cfg.PROCESSOR_ID if hf_cfg else "") or self.model_id
        self.local_dir = local_dir or (hf_cfg.LOCAL_DIR if hf_cfg else None)
        self.trust_remote_code = trust_remote_code if trust_remote_code is not None else (hf_cfg.TRUST_REMOTE_CODE if hf_cfg else False)
        self.use_safetensors = use_safetensors if use_safetensors is not None else (hf_cfg.SAFE_SERIALIZATION if hf_cfg else True)
        self.torch_dtype = self._resolve_torch_dtype(getattr(hf_cfg, "TORCH_DTYPE", "") if hf_cfg else "")
        if self.torch_dtype is None and torch.cuda.is_available():
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.low_cpu_mem_usage = bool(getattr(hf_cfg, "LOW_CPU_MEM_USAGE", False)) if hf_cfg else False
        self.prompt_source, self.system_prompt, self.user_prompt, self.placeholder = self._resolve_prompt_settings(hf_cfg)
        self.use_chat_template = bool(getattr(hf_cfg, "USE_CHAT_TEMPLATE", False)) if hf_cfg else False
        self.attn_implementation = self._resolve_attn_implementation(hf_cfg)
        gen_cfg = hf_cfg.GENERATION if hf_cfg and hasattr(hf_cfg, "GENERATION") else None
        self.generation_kwargs = generation_kwargs or self._resolve_generation_kwargs(gen_cfg)
        mirror = getattr(hf_cfg, "MIRROR", None) if hf_cfg else None
        mirror = mirror or None
        disable_proxy = bool(getattr(hf_cfg, "DISABLE_PROXY", False)) if hf_cfg else False
        allow_unsafe = bool(getattr(hf_cfg, "ALLOW_UNSAFE_TORCH_LOAD", False)) if hf_cfg else False
        self.trainable = bool(getattr(hf_cfg, "TRAINABLE", False)) if hf_cfg else False
        self.lora_cfg = getattr(hf_cfg, "LORA", None) if hf_cfg else None
        self.lora_enabled = bool(getattr(self.lora_cfg, "ENABLED", False)) if self.lora_cfg else False

        cfg_device = hf_cfg.DEVICE if hf_cfg and hasattr(hf_cfg, "DEVICE") else None
        self.device = _get_device(device or cfg_device)

        with _hf_env(mirror, disable_proxy):
            if allow_unsafe:
                self._allow_unsafe_torch_load()
            if self._needs_download():
                self._download_snapshot()
            load_from = self.local_dir if self._local_dir_ready() else self.model_id
            model_kwargs = self._build_model_kwargs()
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_from, trust_remote_code=self.trust_remote_code
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token
            self.tokenizer.padding_side = "left"
            try:
                self.model = self._from_pretrained_with_attn_fallback(
                    AutoModelForCausalLM, load_from, model_kwargs
                )
            except OSError:
                self.model = self._from_pretrained_with_attn_fallback(
                    AutoModelForCausalLM, load_from, self._with_unsafe_safetensors(model_kwargs)
                )

        if hf_cfg and bool(getattr(hf_cfg, "GRADIENT_CHECKPOINTING", False)):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False

        if not self.placeholder:
            self.placeholder = "this is a dummy caption for an undecodable image"
        if not self.use_chat_template and (self.system_prompt or self.user_prompt):
            if hasattr(self.tokenizer, "apply_chat_template"):
                self.use_chat_template = True

        if self.lora_enabled:
            self._apply_lora()
            self.trainable = True

        self.model.to(self.device)
        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, *args, **kwargs):
        if not self.trainable:
            raise NotImplementedError("HFTextCaptionModel is inference-only in this pipeline.")

        input_ids = kwargs.get(cfg.PARAM.INPUT_SENT)
        labels = kwargs.get(cfg.PARAM.TARGET_SENT)
        attention_mask = kwargs.get(cfg.PARAM.ATT_FEATS_MASK)

        model_inputs = {}
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if labels is not None:
            model_inputs["labels"] = labels

        outputs = self.model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if str(cfg.LOSSES.XE_TYPE).lower() == "crossentropy":
            return torch.log_softmax(logits, dim=-1)
        return logits

    def save_lora_adapter(self, output_dir: str) -> bool:
        if not self.lora_enabled:
            return False
        try:
            from peft import PeftModel
        except Exception:
            return False
        if not isinstance(self.model, PeftModel):
            return False
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        return True

    def load_lora_adapter(self, adapter_dir: str) -> bool:
        try:
            from peft import PeftModel
        except Exception:
            return False
        if not os.path.isdir(adapter_dir):
            return False
        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model.to(self.device)
            return True
        except Exception:
            return False

    def _apply_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as exc:
            print(f"[HF] LoRA disabled: peft unavailable ({exc})")
            self.lora_enabled = False
            return
        lora_cfg = self.lora_cfg
        if lora_cfg is None:
            self.lora_enabled = False
            return
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
            self.model = get_peft_model(self.model, lora_config)
        except ValueError as exc:
            print(f"[HF] LoRA target modules not found; disabling LoRA. ({exc})")
            self.lora_enabled = False
            return
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def _local_dir_ready(self) -> bool:
        if not self.local_dir or not os.path.isdir(self.local_dir):
            return False
        has_config = os.path.exists(os.path.join(self.local_dir, "config.json"))
        has_weights = False
        for fname in os.listdir(self.local_dir):
            if fname.startswith("pytorch_model") or fname.endswith(".safetensors"):
                has_weights = True
                break
        return has_config and has_weights

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

    def _resolve_attn_implementation(self, hf_cfg) -> str:
        if not hf_cfg:
            return ""
        value = getattr(hf_cfg, "ATTN_IMPLEMENTATION", "")
        return str(value).strip() if value is not None else ""

    def _strip_attn_implementation(self, model_kwargs: dict) -> dict:
        if "attn_implementation" not in model_kwargs:
            return model_kwargs
        updated = dict(model_kwargs)
        updated.pop("attn_implementation", None)
        return updated

    def _should_retry_without_attn_implementation(self, exc: Exception) -> bool:
        if not self.attn_implementation:
            return False
        msg = str(exc).lower()
        return (
            "attn_implementation" in msg
            or "flash_attention" in msg
            or "flashattention" in msg
            or "flash attention" in msg
        )

    def _from_pretrained_with_attn_fallback(self, model_cls, load_from: str, model_kwargs: dict):
        try:
            return model_cls.from_pretrained(load_from, **model_kwargs)
        except Exception as exc:
            if self._should_retry_without_attn_implementation(exc):
                print("[HF] attn_implementation unsupported; retrying without it.")
                return model_cls.from_pretrained(load_from, **self._strip_attn_implementation(model_kwargs))
            raise

    def _build_model_kwargs(self) -> dict:
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "use_safetensors": self.use_safetensors,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        if self.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        return model_kwargs

    def _with_unsafe_safetensors(self, model_kwargs: dict) -> dict:
        updated = dict(model_kwargs)
        updated["use_safetensors"] = False
        return updated

    def _needs_download(self) -> bool:
        return bool(self.local_dir) and not self._local_dir_ready()

    def _download_snapshot(self) -> None:
        if not self.local_dir:
            return
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            return

        os.makedirs(self.local_dir, exist_ok=True)
        print(f"[HF] Downloading {self.model_id} to {self.local_dir}")
        try:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            print(f"[HF] snapshot_download failed: {exc}. Falling back to cache.")

    def _allow_unsafe_torch_load(self) -> None:
        try:
            from transformers import modeling_utils
            from transformers.utils import import_utils
        except Exception:
            return
        import_utils.check_torch_load_is_safe = lambda: None
        modeling_utils.check_torch_load_is_safe = lambda: None
        print("[HF] WARNING: torch.load safety check disabled (ALLOW_UNSAFE_TORCH_LOAD).")

    def _compose_prompt_text(self) -> str:
        if self.system_prompt and self.user_prompt:
            return f"{self.system_prompt}\n{self.user_prompt}"
        return self.user_prompt or self.system_prompt or ""

    def _build_text_messages(self) -> List[dict]:
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.user_prompt:
            messages.append({"role": "user", "content": self.user_prompt})
        elif self.system_prompt:
            messages.append({"role": "user", "content": ""})
        return messages

    def _prepare_model_inputs(self, count: int) -> dict:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [self._build_text_messages() for _ in range(count)]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
            )
        else:
            prompt_text = self._compose_prompt_text()
            texts = [prompt_text for _ in range(count)]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        return inputs.to(self.device)

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

        count = len(images)
        if count == 0:
            return [], None
        inputs = self._prepare_model_inputs(count)
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["num_beams"] = beam_size

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = self._trim_generated_ids(generated_ids, inputs)
        generated_captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        final_captions = [self.placeholder for _ in range(count)]
        for idx, caption in enumerate(generated_captions):
            final_captions[idx] = caption.strip() if caption.strip() else self.placeholder

        return final_captions, None

    def decode(self, **kwargs):
        kwargs["BEAM_SIZE"] = 1
        return self.decode_beam(**kwargs)
