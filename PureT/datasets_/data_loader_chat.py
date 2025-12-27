import os
import sys
import random
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from lib.config import cfg
import samplers.distributed


class HFCaptionCollator:
    def __init__(
        self,
        processor_id: str,
        trust_remote_code: bool = False,
        use_chat_template: bool = False,
        system_prompt: str = "",
        user_prompt: str = "",
        training_mode: str = "auto",
        max_length: int = 128,
        truncation: bool = True,
        label_ignore: int = -1,
    ) -> None:
        self.processor_id = processor_id
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.training_mode = (training_mode or "auto").lower()
        self.max_length = int(max_length) if max_length else 128
        self.truncation = bool(truncation)
        self.label_ignore = int(label_ignore)

        self._processor = None
        self._pad_token_id = None
        self._is_encoder_decoder = None
        self._model_type = None
        self._warned_truncation = False

    def _ensure_processor(self) -> AutoProcessor:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.processor_id, trust_remote_code=self.trust_remote_code
            )
            self._pad_token_id = getattr(self._processor, "pad_token_id", None)
            if self._pad_token_id is None and hasattr(self._processor, "tokenizer"):
                self._pad_token_id = getattr(self._processor.tokenizer, "pad_token_id", None)
        return self._processor

    def _ensure_is_encoder_decoder(self) -> bool:
        if self._is_encoder_decoder is None:
            try:
                model_cfg = AutoConfig.from_pretrained(
                    self.processor_id, trust_remote_code=self.trust_remote_code
                )
                self._model_type = getattr(model_cfg, "model_type", None)
                self._is_encoder_decoder = bool(getattr(model_cfg, "is_encoder_decoder", False))
            except Exception:
                self._is_encoder_decoder = False
        return self._is_encoder_decoder

    def _is_qwen3_vl(self) -> bool:
        model_type = (self._model_type or "").lower()
        return "qwen3_vl" in model_type

    def _resolve_mode(self) -> str:
        if self.training_mode == "auto":
            return "vision2seq" if self._ensure_is_encoder_decoder() else "chat"
        return self.training_mode

    def _build_chat_text(self, image: Any, caption: str, with_answer: bool) -> str:
        processor = self._ensure_processor()
        if self.use_chat_template and hasattr(processor, "apply_chat_template"):
            messages: List[Dict[str, Any]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            user_content: List[Dict[str, Any]] = []
            if self.user_prompt:
                user_content.append({"type": "text", "text": self.user_prompt})
            user_content.append({"type": "image", "image": image})
            messages.append({"role": "user", "content": user_content})
            if with_answer:
                messages.append({"role": "assistant", "content": caption})
                try:
                    return processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    pass
            try:
                return processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

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
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        user_content: List[Dict[str, Any]] = []
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        if caption is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            )
        return messages

    def _resolve_truncation(self) -> Tuple[bool, Optional[int]]:
        truncation = self.truncation
        max_length = self.max_length
        model_type = (self._model_type or "").lower()
        if truncation and ("qwen" in model_type and "vl" in model_type):
            return truncation, max_length
        return truncation, max_length

    def _safe_processor_call(self, processor, **kwargs):
        try:
            return processor(**kwargs)
        except ValueError as exc:
            msg = str(exc).lower()
            if "image token count" in msg or "mismatch in `image` token count" in msg:
                if not self._warned_truncation:
                    print("[HF] Truncation caused image token mismatch; retrying without truncation.")
                    self._warned_truncation = True
                kwargs["truncation"] = False
                kwargs.pop("max_length", None)
                return processor(**kwargs)
            raise

    def _mask_prompt_labels(
        self,
        labels: Optional[torch.Tensor],
        prompt_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if labels is None or prompt_mask is None or attention_mask is None:
            return
        padding_side = "right"
        processor = self._ensure_processor()
        if hasattr(processor, "tokenizer"):
            padding_side = getattr(processor.tokenizer, "padding_side", "right")
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

    def _safe_apply_chat_template(self, processor, conversation, **kwargs):
        try:
            return processor.apply_chat_template(conversation, **kwargs)
        except ValueError as exc:
            msg = str(exc).lower()
            if "image token count" in msg or "mismatch in `image` token count" in msg:
                if not self._warned_truncation:
                    print("[HF] Truncation caused image token mismatch; retrying without truncation.")
                    self._warned_truncation = True
                kwargs["truncation"] = False
                kwargs.pop("max_length", None)
                return processor.apply_chat_template(conversation, **kwargs)
            raise

    def _select_captions(self, captions: Sequence[str], seq_per_img: int) -> List[str]:
        if not captions:
            return ["."]
        if seq_per_img <= 1:
            return [random.choice(captions)]
        if len(captions) >= seq_per_img:
            return random.sample(captions, seq_per_img)
        repeat_times = seq_per_img // len(captions)
        remainder = seq_per_img % len(captions)
        return list(captions) * repeat_times + list(captions)[:remainder]

    def __call__(self, batch: Sequence[Tuple[Any, ...]]):
        processor = self._ensure_processor()
        self._ensure_is_encoder_decoder()
        indices, captions_list, gv_feat, images = zip(*batch)

        indices = np.stack(indices, axis=0).reshape(-1)
        seq_per_img = max(int(getattr(cfg.DATA_LOADER, "SEQ_PER_IMG", 1)), 1)

        expanded_images: List[Any] = []
        expanded_captions: List[str] = []
        for img, caps in zip(images, captions_list):
            selected = self._select_captions(list(caps), seq_per_img)
            for cap in selected:
                expanded_images.append(img)
                expanded_captions.append(cap)

        mode = self._resolve_mode()
        truncation, max_length = self._resolve_truncation()

        if mode == "vision2seq":
            inputs = self._safe_processor_call(
                processor,
                images=expanded_images,
                text=expanded_captions,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length,
            )
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            labels = input_ids.clone() if input_ids is not None else None
            if labels is not None and self._pad_token_id is not None:
                labels[labels == self._pad_token_id] = self.label_ignore
        else:
            if self.use_chat_template and self._is_qwen3_vl():
                full_messages = [
                    self._build_chat_messages(img, cap)
                    for img, cap in zip(expanded_images, expanded_captions)
                ]
                prompt_messages = [
                    self._build_chat_messages(img, None)
                    for img in expanded_images
                ]
                full_inputs = self._safe_apply_chat_template(
                    processor,
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                    truncation=truncation,
                    max_length=max_length,
                )
                input_ids = full_inputs.get("input_ids")
                attention_mask = full_inputs.get("attention_mask")
                labels = input_ids.clone() if input_ids is not None else None
                if labels is not None and self._pad_token_id is not None:
                    labels[labels == self._pad_token_id] = self.label_ignore

                prompt_inputs = self._safe_apply_chat_template(
                    processor,
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                    truncation=truncation,
                    max_length=max_length,
                )
                prompt_mask = prompt_inputs.get("attention_mask")
                self._mask_prompt_labels(labels, prompt_mask, attention_mask)

                inputs = full_inputs
            else:
                prompt_texts = [
                    self._build_chat_text(img, cap, with_answer=False)
                    for img, cap in zip(expanded_images, expanded_captions)
                ]
                full_texts = [
                    self._build_chat_text(img, cap, with_answer=True)
                    for img, cap in zip(expanded_images, expanded_captions)
                ]

                full_inputs = self._safe_processor_call(
                    processor,
                    text=full_texts,
                    images=expanded_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=truncation,
                    max_length=max_length,
                )
                input_ids = full_inputs.get("input_ids")
                attention_mask = full_inputs.get("attention_mask")
                labels = input_ids.clone() if input_ids is not None else None
                if labels is not None and self._pad_token_id is not None:
                    labels[labels == self._pad_token_id] = self.label_ignore

                prompt_inputs = self._safe_processor_call(
                    processor,
                    text=prompt_texts,
                    images=expanded_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=truncation,
                    max_length=max_length,
                )
                prompt_mask = prompt_inputs.get("attention_mask")
                self._mask_prompt_labels(labels, prompt_mask, attention_mask)

                inputs = full_inputs

        extra_inputs: Dict[str, torch.Tensor] = {
            key: value for key, value in inputs.items() if key not in ("input_ids", "attention_mask")
        }
        att_feats = extra_inputs if extra_inputs else None
        gv_feat_tensor = torch.zeros((input_ids.size(0), 1), dtype=torch.float32)

        return indices, input_ids, labels, gv_feat_tensor, att_feats, attention_mask


def _resolve_hf_processor_id() -> str:
    hf_cfg = getattr(cfg.MODEL, "HF", None)
    if hf_cfg is None:
        raise ValueError("HF config missing; cannot build HF caption loader.")
    model_id = getattr(hf_cfg, "MODEL_ID", "")
    processor_id = getattr(hf_cfg, "PROCESSOR_ID", "") or model_id
    local_dir = getattr(hf_cfg, "LOCAL_DIR", None)
    if local_dir and os.path.isdir(local_dir):
        return local_dir
    return processor_id


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


def load_train(distributed: bool, epoch: int, dataset):
    sampler = samplers.distributed.DistributedSampler(dataset, epoch=epoch) if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False

    num_workers = int(getattr(cfg.DATA_LOADER, "NUM_WORKERS", 0))
    if sys.platform.startswith("win"):
        num_workers = 0
    num_workers = max(0, num_workers)
    pin_memory = bool(getattr(cfg.DATA_LOADER, "PIN_MEMORY", False)) and torch.cuda.is_available()
    persistent_workers = bool(getattr(cfg.DATA_LOADER, "PERSISTENT_WORKERS", True)) and num_workers > 0 and not sys.platform.startswith("win")
    prefetch_factor = int(getattr(cfg.DATA_LOADER, "PREFETCH_FACTOR", 2))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    hf_cfg = getattr(cfg.MODEL, "HF", None)
    processor_id = _resolve_hf_processor_id()
    system_prompt, user_prompt = _resolve_training_prompts(hf_cfg)
    collate_fn = HFCaptionCollator(
        processor_id=processor_id,
        trust_remote_code=bool(getattr(hf_cfg, "TRUST_REMOTE_CODE", False)) if hf_cfg else False,
        use_chat_template=bool(getattr(hf_cfg, "USE_CHAT_TEMPLATE", False)) if hf_cfg else False,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        training_mode=str(getattr(hf_cfg, "TRAIN_MODE", "auto")),
        max_length=int(getattr(hf_cfg, "TRAIN_MAX_LENGTH", 128) or 128),
        truncation=bool(getattr(hf_cfg, "TRAIN_TRUNCATION", True)),
        label_ignore=int(getattr(hf_cfg, "TRAIN_LABEL_IGNORE", -1)),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return loader
