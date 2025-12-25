#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image


def log(msg: str) -> None:
    print(f"[Check] {msg}")


def try_import_modelscope():
    try:
        import modelscope  # noqa: F401
        from modelscope.hub.snapshot_download import snapshot_download  # noqa: F401
        return True
    except Exception as e:
        log(f"ModelScope not available: {e}")
        return False


def ms_snapshot(repo_id: str, local_dir: str) -> Optional[str]:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as e:
        log(f"Cannot import ModelScope snapshot_download: {e}")
        return None
    try:
        os.makedirs(local_dir, exist_ok=True)
        path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        return path
    except Exception as e:
        log(f"ModelScope snapshot_download failed: {e}")
        return None


def check_hf_structure(path: str) -> Dict[str, Any]:
    required = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        # at least one of processor configs
        ("processor_config.json", "preprocessor_config.json"),
        # weights
        ("model.safetensors", "model.safetensors.index.json"),
    ]
    missing = []
    p = Path(path)
    for item in required:
        if isinstance(item, tuple):
            if not any((p / alt).exists() for alt in item):
                missing.append(item)
        else:
            if not (p / item).exists():
                missing.append(item)
    # Extra: if only index exists but shards are missing, flag it
    idx = (p / "model.safetensors.index.json")
    single = (p / "model.safetensors")
    if idx.exists() and not single.exists():
        has_shards = any(p.glob("model-*-of-*.safetensors"))
        if not has_shards:
            missing.append("safetensor_shards")
    return {"path": path, "missing": missing}


def find_nested_hf_dir(root: str) -> Optional[str]:
    """Search for a nested directory that actually contains the shard files.
    Returns the directory path if found, else None.
    """
    root_p = Path(root)
    patterns = ["model-*-of-*.safetensors"]
    # Limit depth to avoid huge scans
    for p in root_p.rglob("*"):
        if not p.is_dir():
            continue
        has_config = (p / "config.json").exists()
        has_index = (p / "model.safetensors.index.json").exists()
        has_shards = any(p.glob(patterns[0]))
        if has_config and has_index and has_shards:
            return str(p)
    return None


def load_hf_components(load_from: str, device_map: str = "auto"):
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    # Tokenizer
    tok_kwargs = dict(trust_remote_code=True, use_fast=False)
    # Try to fix mistral regex if supported
    try:
        tok = AutoTokenizer.from_pretrained(load_from, fix_mistral_regex=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(load_from, **tok_kwargs)
    # Model
    model = AutoModel.from_pretrained(
        load_from,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    # Processor (optional)
    proc = None
    try:
        proc = AutoProcessor.from_pretrained(load_from, trust_remote_code=True)
    except Exception as e:
        log(f"AutoProcessor not available: {e}")
    return model.eval(), tok, proc


def dynamic_preprocess(image: Image.Image, image_size: int = 448, max_num: int = 12):
    # Minimal tiling: resize square only (simple path to avoid heavy logic)
    img = image.convert("RGB").resize((image_size, image_size))
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    pv = transform(img).unsqueeze(0)
    return pv


def run_chat(model, tokenizer, image_path: str) -> Optional[str]:
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = dynamic_preprocess(image).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        question = "<image>\nDescribe the image in detail."
        gen_cfg = dict(max_new_tokens=64, do_sample=False)
        if hasattr(model, "chat"):
            resp = model.chat(tokenizer, pixel_values, question, gen_cfg)
            return resp
        else:
            log("Model has no .chat(); skipping chat test.")
            return None
    except Exception as e:
        log(f"chat() failed: {e}")
        return None


def run_processor_generate(model, processor, image_path: str) -> Optional[str]:
    if processor is None:
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        question = "<image>\nDescribe the image in detail."
        inputs = processor(
            images=[image],
            text=[question],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        # Move tensors to cuda if available
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor) and torch.cuda.is_available():
                inputs[k] = v.cuda()
        outputs = model.generate(**inputs, max_new_tokens=64)
        # Prefer tokenizer if exists
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            text = tok.batch_decode(outputs, skip_special_tokens=True)
            return text[0] if text else None
        return None
    except Exception as e:
        log(f"processor+generate failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Check InternVL loading via ModelScope and HF")
    parser.add_argument("--source", choices=["hf", "ms"], default="hf")
    parser.add_argument("--model", type=str, required=True, help="HF or ModelScope repo id or local path")
    parser.add_argument("--local_dir", type=str, default=None, help="Local dir to load/save model")
    parser.add_argument("--image", type=str, default="PureT/data/coco_karpathy/test_sample_500/00000_id42.jpg")
    args = parser.parse_args()

    load_from = args.model
    if args.source == "ms":
        if os.path.isdir(args.model):
            load_from = args.model
        else:
            if not try_import_modelscope():
                log("Please install ModelScope: pip install modelscope")
                sys.exit(1)
            target_dir = args.local_dir or str(Path.cwd() / (args.model.split('/')[-1]))
            path = ms_snapshot(args.model, target_dir)
            if not path:
                sys.exit(1)
            load_from = path
    else:  # hf
        if args.local_dir and os.path.isdir(args.local_dir):
            load_from = args.local_dir
        elif os.path.isdir(args.model):
            load_from = args.model

    log(f"Loading from: {load_from}")
    struct = check_hf_structure(load_from)
    if struct["missing"]:
        log(f"Potentially missing HF files: {struct['missing']}")
        nested = find_nested_hf_dir(load_from)
        if nested:
            log(f"Detected nested HF folder with shards: {nested}")
            load_from = nested
            struct = check_hf_structure(load_from)
            if struct["missing"]:
                log(f"Still missing in nested folder: {struct['missing']}")

    model, tokenizer, processor = load_hf_components(load_from)
    log("HF components loaded.")

    # Try chat path first
    chat_out = run_chat(model, tokenizer, args.image)
    if chat_out is not None:
        log(f"chat() output: {chat_out}")
    else:
        log("chat() path failed or not available.")

    # Try processor + generate
    gen_out = run_processor_generate(model, processor, args.image)
    if gen_out is not None:
        log(f"processor.generate() output: {gen_out}")
    else:
        log("processor.generate() path failed or not available.")

    if struct["missing"]:
        log("Conversion suggestion: ensure the directory contains config.json, tokenizer files, processor_config.json/preprocessor_config.json, and safetensors (or index). If using ModelScope, point HF loaders to the downloaded directory via --local_dir.")
        log("If tokenizer warning appears (Mistral regex), try fix_mistral_regex=True when loading tokenizer.")


if __name__ == "__main__":
    main()
