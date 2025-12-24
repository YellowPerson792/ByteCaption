#!/usr/bin/env python3
"""Ministral-3-8B-Instruct training launcher (HF Trainer).

This wrapper mirrors the Qwen3-VL training defaults but targets the
Ministral-3-8B-Instruct-2512 weights. Any CLI args you pass will override
the defaults set here.

Example (CLI overrides, aligned with Qwen3-VL header style):
    python tools/train_ministral_hf_trainer.py \
        --folder PureT/experiments/ByteCaption_XE_ministral \
        --dataset coco \
        --model_id mistralai/Ministral-3-8B-Instruct-2512 \
        --processor_id mistralai/Ministral-3-8B-Instruct-2512 \
        --local_dir /autodl-fs/mistralai_Ministral-3-8B-Instruct-2512 \
        --train_samples 0 \
        --val_samples 200 \
        --eval_steps 200 \
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
        --use_hf_defaults \
        --disable_wandb
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

"""
python tools/run_batch_corruption_eval.py --models PureT/experiments/ByteCaption_XE_ministral --corrupt-types rbbf --corrupt-levels S1 S2 S3 S4 S5 --test-samples 0
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_hf_trainer.py"


def _flag_present(argv: List[str], flag: str) -> bool:
    return flag in argv


def _append_default(cmd: List[str], argv: List[str], flag: str, value) -> None:
    if _flag_present(argv, flag):
        return
    cmd.append(flag)
    if isinstance(value, (list, tuple)):
        cmd.extend([str(v) for v in value])
    else:
        cmd.append(str(value))


def _build_command(argv: List[str]) -> List[str]:
    cmd = [sys.executable, str(TRAIN_SCRIPT)]

    _append_default(cmd, argv, "--folder", "PureT/experiments/ByteCaption_XE_ministral")
    _append_default(cmd, argv, "--dataset", "coco")
    _append_default(cmd, argv, "--model_id", "mistralai/Ministral-3-8B-Instruct-2512")
    _append_default(cmd, argv, "--processor_id", "mistralai/Ministral-3-8B-Instruct-2512")
    _append_default(cmd, argv, "--local_dir", "/autodl-fs/data/mistralai_Ministral-3-8B-Instruct-2512")
    _append_default(cmd, argv, "--train_samples", 0)
    _append_default(cmd, argv, "--val_samples", 200)
    _append_default(cmd, argv, "--eval_steps", 200)
    _append_default(cmd, argv, "--best_metric", "SPICE")
    _append_default(cmd, argv, "--early_stop_patience", 4)
    _append_default(cmd, argv, "--max_epoch", 2)
    _append_default(cmd, argv, "--batch_size", 1)
    _append_default(cmd, argv, "--grad_accum_steps", 8)
    _append_default(cmd, argv, "--num_workers", 8)
    _append_default(cmd, argv, "--train_max_length", 512)
    _append_default(cmd, argv, "--train_truncation", 1)
    _append_default(cmd, argv, "--lora_r", 16)
    _append_default(cmd, argv, "--lora_alpha", 32)
    _append_default(cmd, argv, "--lora_dropout", 0.05)
    _append_default(
        cmd,
        argv,
        "--lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    _append_default(cmd, argv, "--attn_implementation", "flash_attention_2")
    if "--use_hf_defaults" not in argv:
        cmd.append("--use_hf_defaults")

    cmd.extend(argv)
    return cmd


def main() -> None:
    argv = sys.argv[1:]
    cmd = _build_command(argv)
    print("[Ministral] Launch:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
