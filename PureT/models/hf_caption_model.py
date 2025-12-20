import os
from contextlib import contextmanager
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
        mirror = getattr(hf_cfg, "MIRROR", None) if hf_cfg else None
        mirror = mirror or None
        disable_proxy = bool(getattr(hf_cfg, "DISABLE_PROXY", False)) if hf_cfg else False
        allow_unsafe = bool(getattr(hf_cfg, "ALLOW_UNSAFE_TORCH_LOAD", False)) if hf_cfg else False

        # Decide device
        cfg_device = hf_cfg.DEVICE if hf_cfg and hasattr(hf_cfg, "DEVICE") else None
        self.device = _get_device(device or cfg_device)

        with _hf_env(mirror, disable_proxy):
            if allow_unsafe:
                self._allow_unsafe_torch_load()
            if self._needs_download():
                self._download_snapshot()
            load_from = self.local_dir if self._local_dir_ready() else self.model_id

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
