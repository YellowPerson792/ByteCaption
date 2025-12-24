from typing import List

from transformers import Qwen3VLForConditionalGeneration

from .hf_caption_model import HFCaptionModel


class Qwen3VLCaptionModel(HFCaptionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_chat_template = True

    def _build_chat_messages(self, image):
        messages = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        user_content = []
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _load_auto_model(self, load_from: str, model_kwargs: dict):
        try:
            return self._from_pretrained_with_attn_fallback(
                Qwen3VLForConditionalGeneration, load_from, model_kwargs
            )
        except Exception:
            return self._from_pretrained_with_attn_fallback(
                Qwen3VLForConditionalGeneration,
                load_from,
                self._with_unsafe_safetensors(model_kwargs),
            )

    def _prepare_model_inputs(self, images: List) -> dict:
        messages = [self._build_chat_messages(image) for image in images]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(self.device)
