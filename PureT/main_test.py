import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler_coco import CocoEvaler
from evaluation.evaler_flickr8k import Flickr8kEvaler
from scorer.coco_scorer import CocoScorer
from scorer.flickr8k_scorer import Flickr8kScorer
from scorer.scorer import Scorer  # 新增导入
from lib.config import cfg, cfg_from_file
from corenet.data.transforms import jpeg_corruption

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

"""
Example:
python PureT/main_test.py --folder PureT/experiments/ByteCaption_XE_openrouter --test_samples 10 --corrupt_types rbbf --corrupt_level S0 --resume -1 --disable_wandb
cd /root/autodl-tmp/ByteCaption && PYTHONPATH=/root/autodl-tmp/ByteCaption python PureT/main_test.py --folder PureT/experiments/ByteCaption_XE_internvl --test_samples 10 --resume -1 --disable_wandb
"""

def _project_root() -> str:
    # main_test.py lives in <project>/PureT/main_test.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_reference_annfile(dataset: str):
    dataset = (dataset or "").lower()
    annfile = None
    if dataset == "coco":
        annfile = getattr(cfg.INFERENCE, "TEST_ANNFILE", None)
    elif dataset == "flickr8k":
        # Flickr8k evaluator in this repo uses VAL_ANNFILE for evaluation
        annfile = getattr(cfg.INFERENCE, "VAL_ANNFILE", None)
    if not annfile:
        return None
    ann_path = annfile
    if not os.path.isabs(ann_path):
        ann_path = os.path.abspath(os.path.join(_project_root(), ann_path))
    return ann_path


def _build_reference_map(ann_path: str):
    """Build image_id -> [reference captions] map from COCO-style annotation JSON."""
    if not ann_path or not os.path.exists(ann_path):
        return None
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
    except Exception:
        return None
    ref_map = {}
    for ann in ann_data.get("annotations", []):
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        if image_id is None or caption is None:
            continue
        try:
            image_id = int(image_id)
        except Exception:
            pass
        ref_map.setdefault(image_id, []).append(caption)
    return ref_map

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args

        self.corrupt_level = jpeg_corruption.normalize_level(getattr(args, "corrupt_level", "S0"))
        self.corrupt_types = getattr(args, "corrupt_types", [])
        # 将 CLI 设置同步到 cfg，供数据加载器读取
        cfg.CORRUPTION.BYTE_STREAM_LEVEL = self.corrupt_level
        cfg.CORRUPTION.BYTE_STREAM_TYPES = self.corrupt_types

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据集类型（与训练脚本保持一致）
        self.dataset_type = getattr(args, "dataset", "coco").lower()

        # 固定随机数种子以便可复现
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(int(cfg.SEED))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cfg.SEED))

        # Evaluation runs are commonly launched as a single process even on multi-GPU hosts.
        # Only enable distributed when torchrun-style environment variables are present.
        self.distributed = (
            torch.distributed.is_available()
            and ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ)
        )

        if self.distributed:
            self.local_rank = init_distributed_mode()
        else:
            self.local_rank = 0

        self.is_master = (not self.distributed) or (dist.is_initialized() and dist.get_rank() == 0)

        # Force safe DataLoader defaults for offline evaluation to avoid process hang on exit.
        # (Common culprit: multi-worker DataLoader + persistent workers.)
        if hasattr(args, "num_workers") and args.num_workers is not None:
            cfg.DATA_LOADER.NUM_WORKERS = max(0, int(args.num_workers))
        if hasattr(args, "pin_memory") and args.pin_memory is not None:
            cfg.DATA_LOADER.PIN_MEMORY = bool(args.pin_memory)

        self._print_eval_summary(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_logging()
        self.setup_wandb()
        self.setup_network()

        # Evalers
        self.val_evaler = None  # not used in test script
        self.test_evaler = None

        # 根据数据集类型创建评估器（与 main.py 对齐），针对 TEST split
        test_samples = getattr(args, "test_samples", 100)
        if test_samples == 0:
            test_samples = None

        enable_eval_loss = True

        # 先不依赖 training_dataset（评估脚本通常不需要训练集引用）
        if self.dataset_type == "coco":
            # 不使用未定义的 self.training_dataset，传 None
            self.scorer = CocoScorer(shared_dataset=None)
            eval_ids_path = cfg.DATA_LOADER.TEST_ID if cfg.DATA_LOADER.TEST_ID else None
            test_annfile = cfg.INFERENCE.TEST_ANNFILE
            gv_feat = getattr(cfg.DATA_LOADER, "TEST_GV_FEAT", cfg.DATA_LOADER.VAL_GV_FEAT)
            att_feats = getattr(cfg.DATA_LOADER, "TEST_ATT_FEATS", cfg.DATA_LOADER.VAL_ATT_FEATS)
            self.test_evaler = CocoEvaler(
                eval_ids_path,
                gv_feat,
                att_feats,
                test_annfile,
                max_samples=test_samples,
                enable_eval_loss=enable_eval_loss,
            )
            self._log(f"Test dataset (COCO): Using {test_samples if test_samples else 'ALL'} samples", prefix="DATASET")
        elif self.dataset_type == "flickr8k":
            self.scorer = Flickr8kScorer(shared_dataset=None)
            eval_ids_path = cfg.DATA_LOADER.VAL_ID if cfg.DATA_LOADER.VAL_ID else None
            val_annfile = cfg.INFERENCE.VAL_ANNFILE
            self.test_evaler = Flickr8kEvaler(
                eval_ids_path,
                cfg.DATA_LOADER.VAL_GV_FEAT,
                cfg.DATA_LOADER.VAL_ATT_FEATS,
                val_annfile,
                max_samples=test_samples,
                enable_eval_loss=enable_eval_loss,
                corrupt_level=self.corrupt_level,
            )
            self._log(f"Test dataset (Flickr8k): Using {test_samples if test_samples else 'ALL'} samples", prefix="DATASET")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # 选择 scorer（仅用于训练时的 reward 计算；评估脚本保留）
        if self.dataset_type == "coco":
            self.scorer = CocoScorer(shared_dataset=None)
        elif self.dataset_type == "flickr8k":
            self.scorer = Flickr8kScorer(shared_dataset=None)
        else:
            self.scorer = Scorer()

    def setup_wandb(self):
        """Initializes wandb if enabled."""
        self.use_wandb = WANDB_AVAILABLE and not self.args.disable_wandb
        if self.is_master and self.use_wandb:
            # Generate a run name if not provided
            if self.args.wandb_name:
                run_name = self.args.wandb_name
            else:
                # e.g., eval-best-corrupt_light
                model_id = 'best' if self.args.resume == -1 else f'epoch_{self.args.resume}'
                run_name = f"eval-{model_id}-corrupt_{self.corrupt_level}"

            wandb.init(
                project=self.args.wandb_project,
                name=run_name,
                config=vars(self.args)  # Log all command-line arguments
            )
            wandb.config.update(cfg, allow_val_change=True) # Log yml config
            self._log("Wandb logging is ENABLED.", prefix="WANDB")
        elif self.is_master:
            self._log("Wandb logging is DISABLED.", prefix="WANDB")

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)

        # Use the original stdout to avoid flush issues on some terminals
        ch = logging.StreamHandler(stream=sys.__stdout__)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        # 避免重复添加handler
        if not self.logger.handlers:
            self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, 'OfflineTest_' + cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)
        if torch.cuda.is_available():
            if not self.distributed and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model = model.to(self.device)
        else:
            self.model = model.to(self.device)

        # 仅在模型不是 HF/BLIP/GIT 时才加载本地检查点。
        model_type = str(getattr(cfg.MODEL, "TYPE", "")).lower()
        is_hf = (
            model_type.startswith("hf")
            or "blip" in model_type
            or "git" in model_type
            or "qwen" in model_type
            or "internvl" in model_type
            or "glm" in model_type
            or "mistral" in model_type
            or "ministral" in model_type
            or "openrouter" in model_type
            or model_type.startswith("gpt")
            or "gpt" in model_type
        )
        if is_hf:
            adapter_dir = None
            if self.args.resume == -1:
                candidate = os.path.join(cfg.ROOT_DIR, "snapshot", "best_lora")
                if os.path.isdir(candidate):
                    adapter_dir = candidate
            elif self.args.resume > 0:
                candidate = os.path.join(cfg.ROOT_DIR, "snapshot", f"lora_{self.args.resume}")
                if os.path.isdir(candidate):
                    adapter_dir = candidate

            if adapter_dir is not None:
                base_model = self.model.module if hasattr(self.model, "module") else self.model
                if hasattr(base_model, "load_lora_adapter"):
                    if base_model.load_lora_adapter(adapter_dir):
                        self.logger.info(f"Loaded LoRA adapter from {adapter_dir}")
                    else:
                        self.logger.info(f"Failed to load LoRA adapter from {adapter_dir}")
            else:
                self.logger.info(f"{cfg.MODEL.TYPE} model uses HF weights. Skipping local checkpoint loading.")
            return

        if self.args.resume > 0:
            ckpt = self.snapshot_path("caption_model", self.args.resume)
            if os.path.exists(ckpt):
                self.model.load_state_dict(
                    torch.load(ckpt, map_location=lambda storage, loc: storage)
                )
                self.logger.info(f"Loaded checkpoint: {ckpt}")
            else:
                self.logger.warning(f"Requested checkpoint for resume not found: {ckpt}")
        elif self.args.resume == -1:
            # 使用 cfg.ROOT_DIR 下 snapshot/best_model.pth，避免硬编码路径
            best_ckpt = os.path.join(cfg.ROOT_DIR or self.args.folder or ".", "snapshot", "best_model.pth")
            if os.path.exists(best_ckpt):
                self.model.load_state_dict(
                    torch.load(best_ckpt, map_location=lambda storage, loc: storage)
                )
                self.logger.info(f"Loaded best model: {best_ckpt}")
            else:
                self.logger.warning(f"best_model.pth not found at: {best_ckpt}")


    def eval(self, epoch):
        # 记录并打印评估结果（仅测试集）
        if self.test_evaler is not None:
            test_res = self.test_evaler(self.model, 'test_' + str(epoch))
            self.logger.info('######## Offline TEST ' + str(epoch) + ' ########')
            self.logger.info(str(test_res))
            if self.is_master and self.use_wandb:
                wandb.log({f"test/{k}": v for k, v in test_res.items()})
            return test_res
        else:
            self.logger.info('TEST evaluation skipped (no test_evaler).')
        return None

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def shutdown(self):
        # Best-effort cleanup to prevent evaluation process hanging on exit.
        try:
            if self.is_master and self.use_wandb:
                wandb.finish()
        except Exception:
            pass

        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        try:
            if hasattr(self, "logger") and self.logger is not None:
                for handler in list(self.logger.handlers):
                    try:
                        handler.flush()
                        handler.close()
                    except Exception:
                        pass
                    try:
                        self.logger.removeHandler(handler)
                    except Exception:
                        pass
        except Exception:
            pass

    def _print_eval_summary(self, args):
        """打印评估相关的摘要（在 Tester 初始化后调用）"""
        if not self.is_master:
            return

        self._log_section("EVALUATION CONFIGURATION")
        print(f"  Dataset Type         : {self.dataset_type.upper()}")
        print(f"  Corrupt Level        : {self.corrupt_level}")
        print(f"  Corrupt Types        : {', '.join(self.corrupt_types) if self.corrupt_types else 'NONE'}")
        resume = getattr(args, "resume", -1)
        mode = "best (--resume==-1)" if resume == -1 else ("checkpoint" if resume > 0 else "auto-latest")
        print(f"  Evaluation Mode      : {mode} (resume={resume})")
        best_path = os.path.join(cfg.ROOT_DIR or args.folder or ".", "snapshot", "best_model.pth")
        print(f"  Best model path      : {best_path}")
        test_samples = getattr(args, "test_samples", 0)
        print(f"  Test Samples         : {test_samples if test_samples > 0 else 'ALL'}")
        # device / distributed
        print(f"  Device               : {self.device}")
        print(f"  Distributed          : {'YES' if self.distributed else 'NO'}")
        # config-based info (best-effort)
        dl_cfg = getattr(cfg, "DATA_LOADER", None)
        num_workers = "N/A"
        if dl_cfg is not None:
            num_workers = getattr(dl_cfg, "NUM_WORKERS", getattr(dl_cfg, "num_workers", "N/A"))
        train_cfg = getattr(cfg, "TRAIN", None)
        batch_size = getattr(train_cfg, "BATCH_SIZE", getattr(train_cfg, "batch_size", "N/A")) if train_cfg is not None else "N/A"
        print(f"  Num workers (data)   : {num_workers}")
        print(f"  Train batch size     : {batch_size}")
        print()

    def _log_section(self, title, width=70):
        """打印带分隔线的章节标题"""
        if not self.is_master:
            return
        print()
        print("=" * width)
        print(f"{title:^{width}}")
        print("=" * width)

    def _log(self, message, level="INFO", prefix=None):
        """统一的日志输出系统"""
        if not self.is_master:
            return
        
        if prefix:
            formatted_message = f"[{prefix}] {message}"
        else:
            formatted_message = f"[{level}] {message}"
        
        print(formatted_message)

def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

def init_distributed_mode():
    # 获取GPU编号
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ["RANK"])
        local_world_size = int(os.environ['WORLD_SIZE'])
        local_gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('[SYSTEM] No distributed environment detected, using single GPU/CPU')
        return 0
    
    torch.cuda.set_device(local_gpu)
    print('[SYSTEM] Distributed init (rank {}): env://'.format(local_rank), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=local_world_size,
        rank=local_rank
    )
    torch.distributed.barrier()
    setup_for_distributed(local_rank==0)
    # 返回GPU编号
    return local_gpu

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning - Offline COCO Test Evaluation')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--resume", type=int, default=-1, help="Checkpoint epoch to load (caption_model_<N>.pth)")
    # Keep backward compatibility with --val_samples
    parser.add_argument("--test_samples", type=int, default=0, help="Number of test samples to use (0 for all)")
    parser.add_argument("--val_samples", type=int, default=None, help="(Alias) Number of test samples to use (0 for all)")
    level_choices = sorted(set(jpeg_corruption.available_levels() + ["none", "light", "medium", "heavy"]))
    parser.add_argument("--corrupt_level", type=str, default="S0", choices=level_choices,
                        help="JPEG bitstream corruption severity (S0-S5/M0-M1; legacy none/light/medium/heavy aliases)")
    parser.add_argument("--corrupt_types", type=str, nargs="+", default=["rbbf"],
                        choices=["rbbf", "rbsl", "metadata_loss", "none"],
                        help="Corruption types to apply to JPEG bitstreams")
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="ByteCaption-Eval", help="Wandb project name for logging.")
    parser.add_argument("--wandb_name", type=str, default=None, help="A specific name for the wandb run.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    # DataLoader options for evaluation (defaults to config values).
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers for evaluation (default: use config).",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory during evaluation.",
    )
    parser.add_argument(
        "--no_pin_memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory during evaluation.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="eval_results/",
        help=(
            "Save evaluation metrics JSON. If this is a directory (recommended), "
            "will write to <dir>/<timestamp>/metrics_<rname>.json. "
            "If this is a file path (ends with .json), will write exactly to that file."
        ),
    )
    parser.add_argument(
        "--no_metrics_out",
        action="store_true",
        help="Disable saving evaluation metrics JSON.",
    )
    parser.add_argument(
        "--save_captions",
        type=int,
        default=-1,
        help=(
            "Save generated captions with references into JSON alongside metrics. "
            "-1 = all evaluated samples, 0 = disable, N>0 = first N samples."
        ),
    )
    parser.set_defaults(pin_memory=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    # Back-compat: allow --val_samples to populate test_samples
    if args.val_samples is not None:
        args.test_samples = args.val_samples
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        # 尝试加载coco数据集的 config
        config_file = 'config_coco.yml'
        config_path = os.path.join(args.folder, config_file)
        if os.path.exists(config_path):
            cfg_from_file(config_path)
            print(f"[CONFIG] Loaded config: {config_path}")
            # 若文件夹名包含 blip，强制切换模型类型，避免默认值覆盖
            if "blip" in args.folder.lower() or os.getenv("FORCE_BLIP", "").lower() in ("1", "true", "yes"):
                cfg.MODEL.TYPE = "BLIP"
                print(f"[CONFIG] FORCE_BLIP active or folder contains 'blip'; cfg.MODEL.TYPE set to BLIP")
            if "git" in args.folder.lower() or os.getenv("FORCE_GIT", "").lower() in ("1", "true", "yes"):
                cfg.MODEL.TYPE = "GIT"
                print(f"[CONFIG] FORCE_GIT active or folder contains 'git'; cfg.MODEL.TYPE set to GIT")
            if "qwen" in args.folder.lower() or os.getenv("FORCE_QWEN", "").lower() in ("1", "true", "yes"):
                cfg.MODEL.TYPE = "QWEN"
                print(f"[CONFIG] FORCE_QWEN active or folder contains 'qwen'; cfg.MODEL.TYPE set to QWEN")
        else:
            # 若找不到，仍尝试读取 generic config.yml（兼容旧项目）
            alt = os.path.join(args.folder, 'config.yml')
            if os.path.exists(alt):
                cfg_from_file(alt)
                print(f"[CONFIG] Loaded config: {alt}")
            else:
                print(f"[WARNING] Config file not found in folder: {args.folder}")

    cfg.ROOT_DIR = args.folder
    tester = Tester(args)

    # --- START: 关键修复 ---
    # 移除所有在 __main__ 块中的模型加载逻辑。
    # Tester 的 __init__ 方法已经通过调用 self.setup_network() 正确处理了模型创建和权重加载。
    # 我们只需要调用 eval 方法即可。

    # 确定要传递给 eval 的 epoch 字符串
    epoch_str = 'best'
    if args.resume > 0:
        epoch_str = str(args.resume)
    
    print(f"\nStarting TEST evaluation for epoch: {epoch_str}")
    metrics = tester.eval(epoch_str)
    # --- END: 关键修复 ---

    # (删除下面所有关于 best_path, latest_ckpt, load_state_dict 和 tester.eval 的旧代码块)
    # if args.resume > 0:
    #     tester.eval(args.resume)
    # elif args.resume == -1:
    #     best_path = ...
    #     if best_path and os.path.exists(best_path):
    #         print(f"Loading best model: {best_path}")
    #         tester.model.load_state_dict(...) # <--- 重复加载导致了问题
    #         tester.eval('best')
    #     else:
    #         ...
    # else:
    #     ...
    
    if metrics is not None and not args.no_metrics_out:
        rname = f"test_{epoch_str}"
        model_folder = args.folder or cfg.ROOT_DIR or os.getcwd()
        model_name = os.path.basename(str(model_folder).rstrip(os.sep))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        corruption_params = {}
        for ctype in args.corrupt_types:
            preset = jpeg_corruption.JPEG_CORRUPTION_PRESETS.get(ctype, {})
            corruption_params[ctype] = preset.get(tester.corrupt_level, {})
        run_record = {
            "model_folder": str(model_folder),
            "model_name": model_name,
            "model_type": str(getattr(cfg.MODEL, "TYPE", "")),
            "corrupt_type": list(args.corrupt_types),
            "corrupt_level": tester.corrupt_level,
            "corruption_params": corruption_params,
            "metrics": metrics,
            "rname": rname,
            "timestamp": timestamp,
        }
        out_spec = args.metrics_out
        if not out_spec:
            out_spec = os.path.join(cfg.ROOT_DIR or model_folder or ".", "eval_results")

        # Treat as directory if:
        # - ends with a path separator, OR
        # - exists and is a directory, OR
        # - has no file extension (common: "eval_results")
        has_ext = bool(os.path.splitext(out_spec)[1])
        is_dir_spec = (
            out_spec.endswith(("/", "\\"))
            or os.path.isdir(out_spec)
            or not has_ext
        )

        if is_dir_spec:
            base_dir = out_spec.rstrip("/\\")
            if not base_dir:
                base_dir = "."
            run_dir = os.path.join(base_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            out_path = os.path.join(run_dir, f"metrics_{rname}.json")
        else:
            out_path = out_spec
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # Save per-sample generations + references (best-effort)
        captions_out_path = None
        captions_count = 0
        if getattr(args, "save_captions", 0) != 0:
            try:
                result_path = os.path.join(str(model_folder), "result", f"result_{rname}.json")
                if os.path.exists(result_path):
                    with open(result_path, "r", encoding="utf-8") as f:
                        raw_results = json.load(f)
                else:
                    raw_results = None

                if isinstance(raw_results, list):
                    max_n = int(getattr(args, "save_captions", -1))
                    if max_n > 0:
                        raw_results = raw_results[:max_n]

                    dataset_type = getattr(tester, "dataset_type", getattr(args, "dataset", "coco"))
                    ann_path = _resolve_reference_annfile(dataset_type)
                    ref_map = _build_reference_map(ann_path) if ann_path else None

                    id_key = getattr(cfg.INFERENCE, "ID_KEY", "image_id")
                    cap_key = getattr(cfg.INFERENCE, "CAP_KEY", "caption")
                    enriched = []
                    for item in raw_results:
                        if not isinstance(item, dict):
                            continue
                        image_id = item.get(id_key)
                        generated = item.get(cap_key)
                        lookup_id = image_id
                        try:
                            lookup_id = int(image_id)
                        except Exception:
                            pass
                        references = []
                        if ref_map is not None:
                            references = ref_map.get(lookup_id) or ref_map.get(image_id) or []
                        enriched.append(
                            {
                                id_key: image_id,
                                cap_key: generated,
                                "references": references,
                            }
                        )

                    captions_count = len(enriched)
                    # place next to metrics output
                    captions_dir = run_dir if is_dir_spec else (os.path.dirname(out_path) or ".")
                    captions_out_path = os.path.join(captions_dir, f"captions_{rname}.json")
                    with open(captions_out_path, "w", encoding="utf-8") as f:
                        json.dump(enriched, f, ensure_ascii=False, indent=2)

                    run_record["references_annfile"] = ann_path
                    run_record["captions_file"] = captions_out_path
                    run_record["captions_count"] = captions_count
            except Exception as exc:
                run_record["captions_error"] = f"{type(exc).__name__}: {exc}"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)
        print(f"[RESULT] Saved metrics JSON: {out_path}")
        if captions_out_path:
            print(f"[RESULT] Saved captions+references JSON: {captions_out_path} (n={captions_count})")

    tester.shutdown()

    # Extra safety: ensure buffered streams are flushed.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
