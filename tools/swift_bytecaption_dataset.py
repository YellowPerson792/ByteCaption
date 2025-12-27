from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from PIL import Image

try:
    from datasets import Features, Image as HFImage, IterableDataset, Value
except Exception:  # pragma: no cover - optional at runtime
    Features = None
    HFImage = None
    IterableDataset = None
    Sequence = None
    Value = None

try:
    from swift.llm.dataset import DatasetMeta, register_dataset
except Exception:  # pragma: no cover - optional at runtime
    DatasetMeta = None
    register_dataset = None

from lib.config import cfg
from PureT.datasets_.coco_dataset import CocoDataset
from PureT.datasets_.flickr8k_dataset import Flickr8kDataset


@dataclass
class SwiftDatasetConfig:
    dataset_type: str
    seq_per_img: int
    train_samples: int
    val_samples: int
    system_prompt: str
    user_prompt: str
    corrupt_types: Optional[List[str]] = None
    corrupt_level: str = "S0"


def _select_captions(captions: List[str], seq_per_img: int) -> List[str]:
    if not captions:
        return ["."]
    if seq_per_img <= 1:
        return [captions[0]]
    if len(captions) >= seq_per_img:
        return list(captions[:seq_per_img])
    repeat_times = seq_per_img // len(captions)
    remainder = seq_per_img % len(captions)
    return list(captions) * repeat_times + list(captions)[:remainder]


def _ensure_image_token(text: str) -> str:
    if "<image>" in text:
        return text
    if text:
        return f"{text}\n<image>"
    return "<image>"


def _maybe_corrupt_image(image: Image.Image, pipeline) -> Image.Image:
    if pipeline is None or not pipeline.is_enabled():
        return image
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        payload = buffer.getvalue()
    except Exception:
        return image

    outputs = pipeline.apply(payload)
    if not outputs:
        return image
    corrupted, _tag = outputs[0]
    try:
        return Image.open(io.BytesIO(corrupted)).convert("RGB")
    except Exception:
        return image


def _build_corruption_pipeline(cfg_data: SwiftDatasetConfig):
    if not cfg_data.corrupt_types:
        return None
    try:
        from corenet.data.transforms.jpeg_corruption import JPEGCorruptionPipeline
    except Exception:
        return None
    return JPEGCorruptionPipeline(
        corruption_types=list(cfg_data.corrupt_types),
        level=str(cfg_data.corrupt_level or "S0"),
    )


def _iter_samples(dataset, cfg_data: SwiftDatasetConfig, max_samples: int) -> Iterable[dict]:
    limit = None if max_samples is None or max_samples <= 0 else max_samples
    image_token_prompt = _ensure_image_token(cfg_data.user_prompt)
    system_prompt = cfg_data.system_prompt
    pipeline = _build_corruption_pipeline(cfg_data)

    count = 0
    for idx in range(len(dataset)):
        _indices, captions, _gv_feat, image = dataset[idx]
        if image is None:
            continue
        if pipeline is not None:
            image = _maybe_corrupt_image(image, pipeline)
        selected = _select_captions(list(captions), cfg_data.seq_per_img)
        for caption in selected:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": image_token_prompt})
            messages.append({"role": "assistant", "content": caption})
            yield {"messages": messages, "images": [image]}
        count += 1
        if limit is not None and count >= limit:
            break


def _build_iterable_dataset(dataset, cfg_data: SwiftDatasetConfig, max_samples: int):
    if IterableDataset is None:
        raise RuntimeError("datasets is unavailable; please install datasets.")

    features = None
    if Features is not None and HFImage is not None and Value is not None:
        features = Features(
            {
                "messages": [{"role": Value("string"), "content": Value("string")}],
                "images": [HFImage()],
            }
        )

    return IterableDataset.from_generator(
        _iter_samples,
        gen_kwargs={"dataset": dataset, "cfg_data": cfg_data, "max_samples": max_samples},
        features=features,
    )


def register_bytecaption_swift_dataset(name: str, cfg_data: SwiftDatasetConfig) -> None:
    if register_dataset is None or DatasetMeta is None:
        raise RuntimeError("swift is unavailable; please install ms-swift.")

    def _load(dataset_syntax, dataset_meta, **kwargs):
        subsets = dataset_syntax.subsets or []
        subset = (subsets[0] if subsets else "train").lower()
        if subset in {"val", "valid", "validation"}:
            split = "val"
        elif subset in {"test", "testing"}:
            split = "test"
        else:
            split = "train"

        if cfg_data.dataset_type == "flickr8k":
            dataset_cls = Flickr8kDataset
        else:
            dataset_cls = CocoDataset

        if split == "val":
            image_ids_path = cfg.DATA_LOADER.VAL_ID
            max_samples = int(cfg_data.val_samples or 0)
            gv_feat_path = getattr(cfg.DATA_LOADER, "VAL_GV_FEAT", cfg.DATA_LOADER.TRAIN_GV_FEAT)
        elif split == "test":
            image_ids_path = cfg.DATA_LOADER.TEST_ID
            max_samples = int(cfg_data.val_samples or 0)
            gv_feat_path = getattr(cfg.DATA_LOADER, "TEST_GV_FEAT", cfg.DATA_LOADER.VAL_GV_FEAT)
        else:
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID
            max_samples = int(cfg_data.train_samples or 0)
            gv_feat_path = getattr(cfg.DATA_LOADER, "TRAIN_GV_FEAT", "")

        dataset = dataset_cls(
            image_ids_path=image_ids_path,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=gv_feat_path,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            max_samples=max_samples if max_samples > 0 else None,
            return_captions=True,
            return_pil=True,
        )
        return _build_iterable_dataset(dataset, cfg_data, max_samples=max_samples)

    dataset_meta = DatasetMeta(dataset_name=name, load_function=_load)
    register_dataset(dataset_meta, exist_ok=True)
