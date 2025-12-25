#!/usr/bin/env python3
"""InternVL3_5-8B LoRA training via ms-swift using ByteCaption datasets.

This script registers a custom ms-swift dataset that reuses the existing
ByteCaption data loader (no JSON export) and launches `swift sft` with
InternVL defaults.

Example (chunked training + small-sample eval; SPICE early-stop handled externally):
  python tools/train_internvl_swift.py \
    --folder PureT/experiments/ByteCaption_XE_internvl \
    --dataset coco \
    --model_dir InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF \
    --model_type internvl_hf \
    --template internvl_hf \
    --output_dir /root/autodl-fs/trained_models/internvl_lora_adapters \
    --train_samples 0 \
    --val_samples 50 \
    --max_epoch 1 \
    --max_steps 200 \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --learning_rate 1e-4 \
    --eval_steps 5 \
    --save_steps 125 \
    --save_total_limit 60 \
    --num_workers 8 \
    --train_max_length 512 \
    --train_system_prompt "You are a vision captioning model." \
    --train_user_prompt "You are given a possibly corrupted JPEG image. Output a short COCO-style caption. Use 5-12 words. Output only the caption with no extra text." \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --attn_impl sdpa \
    --disable_wandb

Note: SPICE-based early-stop should be done by running `tools/run_batch_corruption_eval.py`
on each saved checkpoint (every `--save_steps`) and stopping when SPICE plateaus.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWIFT_ROOT = PROJECT_ROOT / "third_party" / "ms-swift"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "PureT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PureT"))
if str(SWIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIFT_ROOT))

from lib.config import cfg, cfg_from_file  # noqa: E402
from tools.swift_bytecaption_dataset import SwiftDatasetConfig, register_bytecaption_swift_dataset  # noqa: E402


DEFAULT_MODEL_DIR = "InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF"


def parse_args():
    parser = argparse.ArgumentParser(description="ByteCaption InternVL fine-tuning with ms-swift")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "flickr8k"])
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--train_max_length", type=int, default=4096)
    parser.add_argument("--train_system_prompt", type=str, default=None)
    parser.add_argument("--train_user_prompt", type=str, default=None)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["all-linear"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    parser.add_argument("--train_corrupt_types", nargs="+", default=None)
    parser.add_argument("--train_corrupt_level", type=str, default=None)

    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--merged_dir", type=str, default=None)
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--eval_corrupt_types", nargs="+", default=None)
    parser.add_argument("--eval_corrupt_levels", nargs="+", default=None)
    parser.add_argument("--eval_samples", type=int, default=50)

    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def _load_config(folder: Path, dataset: str):
    config_file = "config_coco.yml" if dataset == "coco" else "config_flickr8k.yml"
    config_path = folder / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg_from_file(str(config_path))
    cfg.ROOT_DIR = str(folder)
    return config_path


def _resolve_training_prompts(args) -> Tuple[str, str]:
    hf_cfg = cfg.MODEL.HF
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


def _estimate_max_steps(args, seq_per_img: int) -> Optional[int]:
    if args.max_steps is not None:
        return int(args.max_steps)
    if args.train_samples <= 0:
        return None
    total_rows = int(args.train_samples) * max(int(seq_per_img), 1)
    denom = max(int(args.batch_size), 1) * max(int(args.grad_accum_steps), 1)
    steps_per_epoch = max((total_rows + denom - 1) // denom, 1)
    return int(steps_per_epoch * max(int(args.max_epoch), 1))


def _resolve_model_type(args) -> str:
    if args.model_type:
        return args.model_type
    lowered = str(args.model_dir or "").lower()
    if "internvl3_5" in lowered and "hf" in lowered:
        return "internvl_hf"
    if "internvl" in lowered and "hf" in lowered:
        return "internvl_hf"
    return "internvl3_5"


def _resolve_attn_impl(attn_impl: str) -> str:
    if not attn_impl:
        return "sdpa"
    lowered = attn_impl.lower()
    if "flash" in lowered:
        try:
            import flash_attn  # noqa: F401
        except Exception:
            print("[Swift] flash_attn not available; falling back to sdpa.")
            return "sdpa"
    return attn_impl


def _resolve_run_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if not output_path.exists():
        return output_path

    markers = ("args.json", "trainer_state.json", "logging.jsonl")
    if any((output_path / marker).exists() for marker in markers) or (output_path / "best").exists():
        return output_path

    candidates = [p for p in output_path.iterdir() if p.is_dir() and (p / "args.json").exists()]
    if not candidates:
        candidates = [p for p in output_path.iterdir() if p.is_dir() and (p / "best").exists()]
    if not candidates:
        return output_path
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_swift_args(args, dataset_name: str, max_steps: Optional[int], output_dir: str) -> List[str]:
    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps
    model_type = _resolve_model_type(args)
    template = args.template or model_type
    save_total_limit = max(int(args.save_total_limit), 1)

    swift_args = [
        "--model",
        args.model_dir,
        "--model_type",
        model_type,
        "--template",
        template,
        "--train_type",
        "lora",
        "--dataset",
        f"{dataset_name}:train",
        "--streaming",
        "true",
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--per_device_eval_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.max_epoch),
        "--save_steps",
        str(save_steps),
        "--save_total_limit",
        str(save_total_limit),
        "--max_length",
        str(args.train_max_length),
        "--dataloader_num_workers",
        str(args.num_workers),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        *args.target_modules,
        "--attn_impl",
        _resolve_attn_impl(args.attn_impl),
        "--output_dir",
        output_dir,
        "--load_best_model_at_end",
        "true",
        "--metric_for_best_model",
        "loss",
        "--create_checkpoint_symlink",
        "true",
    ]

    if args.val_samples > 0:
        swift_args.extend(["--val_dataset", f"{dataset_name}:val", "--eval_steps", str(args.eval_steps)])
    else:
        swift_args.extend(["--eval_strategy", "no"])
    if max_steps is not None:
        swift_args.extend(["--max_steps", str(max_steps)])

    if args.disable_wandb:
        swift_args.extend(["--report_to", "none"])

    return swift_args


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    _load_config(folder, args.dataset)

    system_prompt, user_prompt = _resolve_training_prompts(args)
    if args.train_system_prompt is not None:
        system_prompt = args.train_system_prompt
    if args.train_user_prompt is not None:
        user_prompt = args.train_user_prompt

    if args.train_corrupt_types is not None:
        corrupt_types = list(args.train_corrupt_types)
    else:
        corrupt_types = list(getattr(cfg.CORRUPTION, "BYTE_STREAM_TYPES", []) or [])
    if args.train_corrupt_level is not None:
        corrupt_level = str(args.train_corrupt_level)
    else:
        corrupt_level = str(getattr(cfg.CORRUPTION, "BYTE_STREAM_LEVEL", "S0"))

    dataset_name = f"bytecaption_{args.dataset}"
    dataset_cfg = SwiftDatasetConfig(
        dataset_type=args.dataset,
        seq_per_img=int(getattr(cfg.DATA_LOADER, "SEQ_PER_IMG", 1)),
        train_samples=int(args.train_samples),
        val_samples=int(args.val_samples),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        corrupt_types=corrupt_types,
        corrupt_level=corrupt_level,
    )
    register_bytecaption_swift_dataset(dataset_name, dataset_cfg)

    output_dir = args.output_dir or str(folder / "snapshot" / "swift_lora")
    os.makedirs(output_dir, exist_ok=True)

    max_steps = _estimate_max_steps(args, dataset_cfg.seq_per_img)
    if max_steps is None:
        raise ValueError("max_steps is required when train_samples=0 in streaming mode")

    swift_args = _build_swift_args(args, dataset_name, max_steps=max_steps, output_dir=output_dir)

    from swift.llm.train.sft import sft_main

    print("[Swift] Launching ms-swift SFT:", " ".join(swift_args))
    sft_main(swift_args)

    if not args.merge_lora:
        return

    from swift.llm.export.export import export_main

    run_dir = _resolve_run_dir(output_dir)
    adapter_dir = run_dir / "best" if (run_dir / "best").exists() else run_dir
    merged_dir = args.merged_dir or f"{adapter_dir}-merged"
    export_args = [
        "--model",
        args.model_dir,
        "--adapters",
        str(adapter_dir),
        "--merge_lora",
        "true",
        "--output_dir",
        str(merged_dir),
        "--safe_serialization",
        "true",
    ]
    print("[Swift] Merging LoRA:", " ".join(export_args))
    export_main(export_args)

    if not args.run_eval:
        return

    eval_folder = folder / "snapshot" / "internvl_swift_eval"
    os.makedirs(eval_folder, exist_ok=True)
    base_cfg_path = folder / ("config_coco.yml" if args.dataset == "coco" else "config_flickr8k.yml")
    eval_cfg_path = eval_folder / base_cfg_path.name

    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    cfg_text = re.sub(
        r"LOCAL_DIR:\s*['\"]?[^'\"]*['\"]?",
        f"LOCAL_DIR: '{merged_dir}'",
        cfg_text,
    )
    with open(eval_cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_text)

    corrupt_types = args.eval_corrupt_types or ["rbbf", "rbsl", "metadata_loss"]
    corrupt_levels = args.eval_corrupt_levels or ["S0", "S1", "S2", "S3", "S4", "S5"]
    eval_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_batch_corruption_eval.py"),
        "--models",
        str(eval_folder),
        "--corrupt-types",
        *corrupt_types,
        "--corrupt-levels",
        *corrupt_levels,
        "--test-samples",
        str(args.eval_samples),
        "--dataset",
        args.dataset,
    ]
    print("[Swift] Running corruption eval:", " ".join(eval_cmd))
    os.system(" ".join(eval_cmd))


if __name__ == "__main__":
    main()
