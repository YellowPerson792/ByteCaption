import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from lib.config import cfg


def _get_device(cfg_device: Optional[str]) -> torch.device:
    if cfg_device and cfg_device.lower() != "auto":
        return torch.device(cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFCaptionModel(nn.Module):
    """
    Generic HuggingFace vision captioning wrapper (BLIP-compatible).
    Accepts a list of PIL images (with optional None placeholders) via cfg.PARAM.ATT_FEATS.
    """

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
        self.model_id = model_id or (hf_cfg.MODEL_ID if hf_cfg else "Salesforce/blip-image-captioning-base")
        self.processor_id = processor_id or (hf_cfg.PROCESSOR_ID if hf_cfg else "") or self.model_id
        self.local_dir = local_dir or (hf_cfg.LOCAL_DIR if hf_cfg else None)
        self.trust_remote_code = trust_remote_code if trust_remote_code is not None else (hf_cfg.TRUST_REMOTE_CODE if hf_cfg else False)
        self.use_safetensors = use_safetensors if use_safetensors is not None else (hf_cfg.SAFE_SERIALIZATION if hf_cfg else True)
        gen_cfg = hf_cfg.GENERATION if hf_cfg and hasattr(hf_cfg, "GENERATION") else None
        self.generation_kwargs = generation_kwargs or {
            "max_length": gen_cfg.MAX_LENGTH if gen_cfg else 50,
            "num_beams": gen_cfg.NUM_BEAMS if gen_cfg else 3,
        }

        # Decide device
        cfg_device = hf_cfg.DEVICE if hf_cfg and hasattr(hf_cfg, "DEVICE") else None
        self.device = _get_device(device or cfg_device)

        load_from = self.local_dir if self.local_dir and os.path.exists(self.local_dir) else self.model_id

        # Prefer BLIP-specific classes when the model id looks like BLIP for better compatibility
        if "blip" in self.model_id.lower():
            self.processor = BlipProcessor.from_pretrained(
                load_from, trust_remote_code=self.trust_remote_code
            )
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    load_from,
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=self.use_safetensors,
                )
            except OSError:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    load_from,
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=False,
                )
        else:
            self.processor = AutoProcessor.from_pretrained(
                load_from, trust_remote_code=self.trust_remote_code
            )
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    load_from,
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=self.use_safetensors,
                )
            except OSError:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    load_from,
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=False,
                )

        self.model.to(self.device)
        self.model.eval()

    def forward(self, *args, **kwargs):
        # Not used during evaluation; placeholder to satisfy nn.Module API
        raise NotImplementedError("HFCaptionModel is inference-only in this pipeline.")

    def _prepare_inputs(self, images: Sequence) -> Tuple[List[int], List]:
        valid_images_with_indices = [(i, img) for i, img in enumerate(images) if img is not None]
        if not valid_images_with_indices:
            return [], []
        original_indices, valid_images = zip(*valid_images_with_indices)
        return list(original_indices), list(valid_images)

    def decode_beam(self, **kwargs):
        images = kwargs[cfg.PARAM.ATT_FEATS]
        beam_size = kwargs.get("BEAM_SIZE", self.generation_kwargs.get("num_beams", 3))

        original_indices, valid_images = self._prepare_inputs(images)
        dummy_caption = "this is a dummy caption for an undecodable image"

        if not valid_images:
            return [dummy_caption for _ in range(len(images))], None

        inputs = self.processor(images=valid_images, return_tensors="pt").to(self.device)
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["num_beams"] = beam_size

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        final_captions = [dummy_caption for _ in range(len(images))]
        for idx, caption in zip(original_indices, generated_captions):
            final_captions[idx] = caption.strip() if caption.strip() else dummy_caption

        return final_captions, None

    def decode(self, **kwargs):
        kwargs["BEAM_SIZE"] = 1
        return self.decode_beam(**kwargs)
