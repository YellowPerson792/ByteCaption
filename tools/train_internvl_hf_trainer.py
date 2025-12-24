#!/usr/bin/env python3
"""InternVL3_5-8B training launcher (HF Trainer).

This wrapper mirrors the Qwen3/Ministral defaults but targets
OpenGVLab/InternVL3_5-8B. Any CLI args you pass will override the defaults.
It also auto-downloads weights into the working directory using hf-mirror,
with HTTP proxies disabled, and backs up to /root/autodl-fs.

Example:
    python tools/train_internvl_hf_trainer.py \
        --folder PureT/experiments/ByteCaption_XE_internvl \
        --dataset coco \
        --model_id OpenGVLab/InternVL3_5-8B \
        --processor_id OpenGVLab/InternVL3_5-8B \
        --local_dir ./InternVL3_5-8B \
        --train_samples 0 \
        --val_samples 5 \
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

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_hf_trainer.py"

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-8B"
DEFAULT_LOCAL_DIR = str(PROJECT_ROOT / "InternVL3_5-8B")
DEFAULT_FOLDER = "PureT/experiments/ByteCaption_XE_internvl"
DEFAULT_MIRROR = "https://hf-mirror.com"
BACKUP_ROOT = "/root/autodl-fs"


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


def _flag_present(argv: List[str], flag: str) -> bool:
    return flag in argv


def _get_flag_value(argv: List[str], flag: str) -> Optional[str]:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def _append_default(cmd: List[str], argv: List[str], flag: str, value) -> None:
    if _flag_present(argv, flag):
        return
    cmd.append(flag)
    if isinstance(value, (list, tuple)):
        cmd.extend([str(v) for v in value])
    else:
        cmd.append(str(value))


def _snapshot_ready(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_weights = False
    for fname in os.listdir(path):
        if fname.startswith("pytorch_model") or fname.endswith(".safetensors"):
            has_weights = True
            break
    return has_config and has_weights


def _copy_tree(src: str, dst: str) -> None:
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dest_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(dest_root, exist_ok=True)
        for fname in files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dest_root, fname)
            if os.path.exists(dst_path):
                try:
                    if os.path.getsize(dst_path) == os.path.getsize(src_path):
                        continue
                except OSError:
                    pass
            shutil.copy2(src_path, dst_path)


def _ensure_weights(model_id: str, local_dir: str) -> None:
    if not local_dir:
        return
    if not _snapshot_ready(local_dir):
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            print(f"[InternVL] huggingface_hub unavailable: {exc}")
            return

        os.makedirs(local_dir, exist_ok=True)
        print(f"[InternVL] Downloading {model_id} to {local_dir}")
        with _hf_env(DEFAULT_MIRROR, True):
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )

    backup_name = os.path.basename(os.path.normpath(local_dir))
    if not backup_name:
        return
    backup_dir = os.path.join(BACKUP_ROOT, backup_name)
    if _snapshot_ready(backup_dir):
        return
    os.makedirs(backup_dir, exist_ok=True)
    print(f"[InternVL] Backing up weights to {backup_dir}")
    _copy_tree(local_dir, backup_dir)


def _build_command(argv: List[str]) -> List[str]:
    cmd = [sys.executable, str(TRAIN_SCRIPT)]

    _append_default(cmd, argv, "--folder", DEFAULT_FOLDER)
    _append_default(cmd, argv, "--dataset", "coco")
    _append_default(cmd, argv, "--model_id", DEFAULT_MODEL_ID)
    _append_default(cmd, argv, "--processor_id", DEFAULT_MODEL_ID)
    _append_default(cmd, argv, "--local_dir", DEFAULT_LOCAL_DIR)
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
    model_id = _get_flag_value(argv, "--model_id") or DEFAULT_MODEL_ID
    local_dir = _get_flag_value(argv, "--local_dir") or DEFAULT_LOCAL_DIR

    _ensure_weights(model_id, local_dir)

    cmd = _build_command(argv)
    print("[InternVL] Launch:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
