import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "PureT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

from lib.config import cfg, cfg_from_file  # noqa: E402
from PureT.main import Trainer  # noqa: E402


"""Example:
python tools/train_hf_lora.py --folder PureT/experiments/ByteCaption_XE_qwen --dataset coco \
  --model_id Qwen/Qwen3-VL-8B-Instruct --processor_id Qwen/Qwen3-VL-8B-Instruct --local_dir ./Qwen3-VL-8B-Instruct \
  --train_samples 0 --val_samples 20 --eval_steps 100 --best_metric SPICE --max_epoch 2 --test_interval 999 --train_max_length 1024\
  --batch_size 1 --grad_accum_steps 8 --num_workers 8 --disable_wandb
  
"""


def parse_args():
    parser = argparse.ArgumentParser(description="HF LoRA caption training (reuse main.py loop)")
    parser.add_argument("--folder", type=str, required=True, help="Experiment folder (contains config_*.yml)")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "flickr8k"])
    parser.add_argument("--resume", type=int, default=-2)
    parser.add_argument("--load_epoch", action="store_true")
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=40)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--best_metric", type=str, default="SPICE")
    parser.add_argument("--keep_full_metrics", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ByteCaption")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--processor_id", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default=None, choices=["auto", "vision2seq", "chat"])
    parser.add_argument("--train_system_prompt", type=str, default=None)
    parser.add_argument("--train_user_prompt", type=str, default=None)
    parser.add_argument("--train_max_length", type=int, default=None)
    parser.add_argument("--train_truncation", type=int, default=None, choices=[0, 1])

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", type=str, default=None)
    parser.add_argument("--lora_task_type", type=str, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    parser.add_argument("--lora_modules_to_save", nargs="+", default=None)
    parser.add_argument("--lora_save_full_model", action="store_true")

    parser.add_argument("--seq_per_img", type=int, default=None)
    parser.add_argument("--shuffle", type=int, default=None, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", type=int, default=None, choices=[0, 1])
    parser.add_argument("--prefetch_factor", type=int, default=None)

    parser.add_argument("--skip_corruption_eval", action="store_true")
    parser.add_argument("--corrupt_types", nargs="+", default=None)
    parser.add_argument("--corrupt_levels", nargs="+", default=None)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--test_interval", type=int, default=None)
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)

    # Debug: print actual HF model inputs (shapes + decoded text preview)
    parser.add_argument("--debug_print_inputs", action="store_true", help="Print HF model inputs for the first training forward")
    parser.add_argument("--debug_print_inputs_every", type=int, default=0, help="If >0, print every N forwards (0=only once)")
    parser.add_argument("--debug_print_inputs_max_tokens", type=int, default=256, help="Max tokens to decode for preview")
    return parser.parse_args()


def _load_config(folder: Path, dataset: str):
    config_file = "config_coco.yml" if dataset == "coco" else "config_flickr8k.yml"
    config_path = folder / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg_from_file(str(config_path))
    cfg.ROOT_DIR = str(folder)
    return config_path


def _apply_hf_overrides(args):
    hf_cfg = cfg.MODEL.HF
    hf_cfg.TRAINABLE = True
    hf_cfg.LORA.ENABLED = True

    if args.model_id:
        hf_cfg.MODEL_ID = args.model_id
    if args.processor_id:
        hf_cfg.PROCESSOR_ID = args.processor_id
    if args.local_dir:
        hf_cfg.LOCAL_DIR = args.local_dir
    if args.train_mode:
        hf_cfg.TRAIN_MODE = args.train_mode
    if args.train_system_prompt is not None:
        hf_cfg.TRAIN_SYSTEM_PROMPT = args.train_system_prompt
    if args.train_user_prompt is not None:
        hf_cfg.TRAIN_USER_PROMPT = args.train_user_prompt
    if args.train_max_length is not None:
        hf_cfg.TRAIN_MAX_LENGTH = int(args.train_max_length)
    if args.train_truncation is not None:
        hf_cfg.TRAIN_TRUNCATION = bool(args.train_truncation)

    if args.lora_r is not None:
        hf_cfg.LORA.R = int(args.lora_r)
    if args.lora_alpha is not None:
        hf_cfg.LORA.ALPHA = int(args.lora_alpha)
    if args.lora_dropout is not None:
        hf_cfg.LORA.DROPOUT = float(args.lora_dropout)
    if args.lora_bias is not None:
        hf_cfg.LORA.BIAS = args.lora_bias
    if args.lora_task_type is not None:
        hf_cfg.LORA.TASK_TYPE = args.lora_task_type
    if args.lora_target_modules is not None:
        hf_cfg.LORA.TARGET_MODULES = args.lora_target_modules
    if args.lora_modules_to_save is not None:
        hf_cfg.LORA.MODULES_TO_SAVE = args.lora_modules_to_save
    if args.lora_save_full_model:
        hf_cfg.LORA.SAVE_FULL_MODEL = True
    if args.gradient_checkpointing:
        hf_cfg.GRADIENT_CHECKPOINTING = True
    if args.attn_implementation is not None:
        hf_cfg.ATTN_IMPLEMENTATION = args.attn_implementation


def _apply_dataloader_overrides(args):
    if args.seq_per_img is not None:
        cfg.DATA_LOADER.SEQ_PER_IMG = int(args.seq_per_img)
    else:
        cfg.DATA_LOADER.SEQ_PER_IMG = 1
    if args.shuffle is not None:
        cfg.DATA_LOADER.SHUFFLE = bool(args.shuffle)
    else:
        cfg.DATA_LOADER.SHUFFLE = True
    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_WORKERS = int(args.num_workers)
    if args.pin_memory is not None:
        cfg.DATA_LOADER.PIN_MEMORY = bool(args.pin_memory)
    if args.prefetch_factor is not None:
        cfg.DATA_LOADER.PREFETCH_FACTOR = int(args.prefetch_factor)


def _apply_solver_overrides(args):
    if args.max_epoch is not None:
        cfg.SOLVER.MAX_EPOCH = int(args.max_epoch)
    if args.test_interval is not None:
        cfg.SOLVER.TEST_INTERVAL = int(args.test_interval)
    if args.base_lr is not None:
        cfg.SOLVER.BASE_LR = float(args.base_lr)
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = int(args.batch_size)
    if args.test_batch_size is not None:
        cfg.TEST.BATCH_SIZE = int(args.test_batch_size)
    if args.grad_accum_steps is not None:
        args.grad_accum_steps = int(args.grad_accum_steps)


def _auto_tune_for_hf(args):
    model_id = str(getattr(cfg.MODEL.HF, "MODEL_ID", "")).lower()
    is_qwen_vl = "qwen" in model_id and "vl" in model_id
    if is_qwen_vl:
        if args.batch_size is None and cfg.TRAIN.BATCH_SIZE > 1:
            cfg.TRAIN.BATCH_SIZE = 1
            print("[HF] Auto-setting TRAIN.BATCH_SIZE=1 for Qwen-VL. Use --batch_size to override.")
        if args.test_batch_size is None and cfg.TEST.BATCH_SIZE > 1:
            cfg.TEST.BATCH_SIZE = 1
            print("[HF] Auto-setting TEST.BATCH_SIZE=1 for Qwen-VL. Use --test_batch_size to override.")


def _run_corruption_eval(args):
    if args.skip_corruption_eval:
        return
    corrupt_types = args.corrupt_types or ["rbbf", "rbsl", "metadata_loss"]
    corrupt_levels = args.corrupt_levels or ["S0", "S1", "S2", "S3", "S4", "S5"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_batch_corruption_eval.py"),
        "--models",
        str(Path(args.folder)),
        "--corrupt-types",
        *corrupt_types,
        "--corrupt-levels",
        *corrupt_levels,
        "--test-samples",
        str(args.test_samples),
        "--dataset",
        args.dataset,
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    if args.debug_print_inputs:
        os.environ["BYTECAPTION_DEBUG_INPUTS"] = "1"
        os.environ["BYTECAPTION_DEBUG_INPUTS_ONCE"] = "1" if (args.debug_print_inputs_every or 0) <= 0 else "0"
        os.environ["BYTECAPTION_DEBUG_INPUTS_EVERY"] = str(int(args.debug_print_inputs_every or 0))
        os.environ["BYTECAPTION_DEBUG_INPUTS_MAX_TOKENS"] = str(int(args.debug_print_inputs_max_tokens or 256))

    folder = Path(args.folder)
    _load_config(folder, args.dataset)

    _apply_hf_overrides(args)
    _apply_dataloader_overrides(args)
    _apply_solver_overrides(args)
    _auto_tune_for_hf(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(args)
    trainer.train()
    _run_corruption_eval(args)


if __name__ == "__main__":
    main()
