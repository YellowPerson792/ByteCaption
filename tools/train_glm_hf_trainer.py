#!/usr/bin/env python3
"""GLM-4.6V-Flash training launcher (HF Trainer).

This wrapper mirrors the Qwen/Ministral defaults but targets
zai-org/GLM-4.6V-Flash. Any CLI args you pass will override defaults.

Example:
    python tools/train_glm_hf_trainer.py \
        --folder PureT/experiments/ByteCaption_XE_glm \
        --dataset coco \
        --model_id zai-org/GLM-4.6V-Flash \
        --processor_id zai-org/GLM-4.6V-Flash \
        --local_dir ./GLM-4.6V-Flash \
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
from typing import List

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

    _append_default(cmd, argv, "--folder", "PureT/experiments/ByteCaption_XE_glm")
    _append_default(cmd, argv, "--dataset", "coco")
    _append_default(cmd, argv, "--model_id", "zai-org/GLM-4.6V-Flash")
    _append_default(cmd, argv, "--processor_id", "zai-org/GLM-4.6V-Flash")
    _append_default(cmd, argv, "--local_dir", "./GLM-4.6V-Flash")
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
    print("[GLM] Launch:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
