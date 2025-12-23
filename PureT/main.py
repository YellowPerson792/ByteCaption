"""Main training entry (ByteFormer Captioning) supporting COCO & Flickr8k.
"""

import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb
WANDB_AVAILABLE = True

import losses
import models

# COCO 组件
from datasets_.coco_dataset_hf import CocoDataset
from datasets_.data_loader_byteformer_coco import load_train as load_train_coco
from datasets_.data_loader_hf_caption import load_train as load_train_hf
from evaluation.evaler_coco import CocoEvaler
from scorer.coco_scorer import CocoScorer

# Flickr8k 组件
from datasets_.flickr8k_dataset_hf import Flickr8kDataset
from datasets_.data_loader_byteformer_f8k import load_train as load_train_f8k
from evaluation.evaler_flickr8k import Flickr8kEvaler
from scorer.flickr8k_scorer import Flickr8kScorer

import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file


"""
cd /d/MLLMs/ByteCaption && python PureT/main.py --folder PureT/experiments/ByteCaption_XE --eval_steps 100 --dataset coco --freeze_backbone --disable_wandb
cd /root/autodl-tmp/ByteCaption && PYTHONPATH=/root/autodl-tmp/ByteCaption python PureT/main.py --folder PureT/experiments/ByteCaption_XE --dataset coco --eval_steps 300 --early_stop_patience 4 --val_samples 100 --load_weights --freeze_backbone  --disable_wandb
cd /root/autodl-tmp/ByteCaption && PYTHONPATH=/root/autodl-tmp/ByteCaption torchrun --nproc_per_node=2 --master_port=12355 PureT/main.py --folder PureT/experiments/ByteCaption_XE --eval_steps 300 --val_samples 50 --dataset coco --load_weights --freeze_backbone
"""

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        
        # 获取数据集选择参数
        self.dataset_type = getattr(args, 'dataset', 'coco').lower()
        self.best_metric = getattr(args, 'best_metric', 'CIDEr')
        grad_steps = getattr(args, 'grad_accum_steps', 1)
        if grad_steps is None:
            grad_steps = 1
        self.grad_accum_steps = max(1, int(grad_steps))
        
        # 设置随机数种子
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed(cfg.SEED)

        # Check if distributed training should be used
        self.distributed = torch.cuda.device_count() > 1 and torch.distributed.is_available()
        
        if self.distributed:
            self.local_rank = init_distributed_mode()
        else:
            self.local_rank = 0
        self.is_master = (not self.distributed) or (dist.get_rank() == 0 if self.distributed else True)

        # 显示数据集配置 (需要在is_master初始化后)
        self._print_config_summary(args)        # Choose device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练脚本默认跳过 METEOR/SPICE 等耗时指标（评估脚本保持原样）
        self._maybe_disable_slow_metrics()

        # SCST标记
        self.rl_stage = False
        # 设置日志写入
        self.setup_logging()
        # 设置wandb
        self.setup_wandb()
        # 训练数据集
        self.setup_dataset()
       
        self.setup_evaler()
        
        # 训练模型结构
        self.setup_network()
        
        # Use appropriate scorer based on dataset type
        if self.dataset_type == 'coco':
            self.scorer = CocoScorer(shared_dataset=self.training_dataset)
        elif self.dataset_type == 'flickr8k':
            self.scorer = Flickr8kScorer(shared_dataset=self.training_dataset)
        else:
            self.scorer = Scorer()

        # 初始化早停和最佳分数变量（基于 best_metric）
        self.best_cider = float('-inf')
        self.best_epoch = -1
        self.best_step = None
        self.evals_since_improvement = 0
        self.early_stop_patience = getattr(args, 'early_stop_patience', 0)
        self.latest_val_res = None
            
    def setup_evaler(self):
        # 使用命令行参数或默认值
        val_samples = getattr(self.args, 'val_samples', 100)
        if val_samples == 0:
            val_samples = None  # None表示使用所有样本

        # 默认启用eval_loss功能
        enable_eval_loss = True

        # 根据数据集类型设置评估器
        if self.dataset_type == 'coco':
            eval_ids_path = cfg.DATA_LOADER.VAL_ID if cfg.DATA_LOADER.VAL_ID else None
            val_annfile = cfg.INFERENCE.VAL_ANNFILE
            
            self.val_evaler = CocoEvaler(
                eval_ids_path,
                cfg.DATA_LOADER.VAL_GV_FEAT,
                cfg.DATA_LOADER.VAL_ATT_FEATS,
                val_annfile,
                max_samples=val_samples,
                enable_eval_loss=enable_eval_loss
            )
            self._log(f"Validation dataset (COCO): Using {val_samples if val_samples else 'ALL'} samples", prefix="DATASET")
        elif self.dataset_type == 'flickr8k':
            eval_ids_path = cfg.DATA_LOADER.VAL_ID if cfg.DATA_LOADER.VAL_ID else None
            val_annfile = cfg.INFERENCE.VAL_ANNFILE
            
            self.val_evaler = Flickr8kEvaler(
                eval_ids_path,
                cfg.DATA_LOADER.VAL_GV_FEAT,
                cfg.DATA_LOADER.VAL_ATT_FEATS,
                val_annfile,
                max_samples=val_samples,
                enable_eval_loss=enable_eval_loss
            )
            self._log(f"Validation dataset (Flickr8k): Using {val_samples if val_samples else 'ALL'} samples", prefix="DATASET")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        self._log("XE Loss during evaluation: ENABLED (default)", prefix="DATASET")
        
        # 暂时去掉测试集评估以节省调试时间
        self.test_evaler = None
    
        # 训练数据集导入
    def setup_dataset(self):
        # 使用命令行参数或默认值
        train_samples = getattr(self.args, 'train_samples', 200)
        if train_samples == 0:
            train_samples = None  # None表示使用所有样本

        # 使用生成的 JSON 文件进行训练
        train_id_path = cfg.DATA_LOADER.TRAIN_ID if cfg.DATA_LOADER.TRAIN_ID else None
        
        if self.dataset_type == 'coco':
            self.training_dataset = CocoDataset(
                image_ids_path=train_id_path,
                input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
                target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
                gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
                seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
                max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
                max_samples=train_samples,
                return_captions=self._is_hf_training(),
                return_pil=self._is_hf_training(),
            )
            self._log(f"Training dataset (COCO): Using {train_samples if train_samples else 'ALL'} samples", prefix="DATASET")
        elif self.dataset_type == 'flickr8k':
            self.training_dataset = Flickr8kDataset(
                image_ids_path=train_id_path,
                input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
                target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
                gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
                seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
                max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
                max_samples=train_samples,
                return_captions=self._is_hf_training(),
                return_pil=self._is_hf_training(),
            )
            self._log(f"Training dataset (Flickr8k): Using {train_samples if train_samples else 'ALL'} samples", prefix="DATASET")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # 为了向后兼容，保留coco_set引用
        self.coco_set = self.training_dataset

    # DataLoader
    def setup_loader(self, epoch):
        if self.dataset_type == 'coco':
            if self._is_hf_training():
                self.training_loader = load_train_hf(
                    self.distributed, epoch, self.training_dataset)
            else:
                self.training_loader = load_train_coco(
                    self.distributed, epoch, self.training_dataset)
        elif self.dataset_type == 'flickr8k':
            if self._is_hf_training():
                self.training_loader = load_train_hf(
                    self.distributed, epoch, self.training_dataset)
            else:
                self.training_loader = load_train_f8k(
                    self.distributed, epoch, self.training_dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    # 设置日志写入
    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        # 使用多卡训练时不输出日志
        if self.distributed and dist.get_rank() > 0:
            return

        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        
        """
        # 日志的屏幕打印
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        """

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def _log(self, message, level="INFO", prefix=None):
        """统一的日志输出系统"""
        if not self.is_master:
            return
        
        if prefix:
            formatted_message = f"[{prefix}] {message}"
        else:
            formatted_message = f"[{level}] {message}"
        
        print(formatted_message)
    
    def _log_section(self, title, width=70):
        """打印带分隔线的章节标题"""
        if not self.is_master:
            return
        print()
        print("=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    def _log_subsection(self, title, width=50):
        """打印子章节标题"""
        if not self.is_master:
            return
        print(f"\n{'-' * width}")
        print(f"{title}")
        print(f"{'-' * width}")

    def _best_marker_label(self):
        """返回当前最佳指标所在的位置描述"""
        if self.best_step is not None:
            return f"step {self.best_step}"
        if self.best_epoch > 0:
            return f"epoch {self.best_epoch}"
        return "N/A"

    def _print_config_summary(self, args):
        """打印配置摘要"""
        if not self.is_master:
            return
            
        train_samples = getattr(args, 'train_samples', 200)
        val_samples = getattr(args, 'val_samples', 100)
        eval_steps = getattr(args, 'eval_steps', 0)
        log_steps = getattr(args, 'log_steps', 40)
        freeze_backbone = getattr(args, 'freeze_backbone', False)
        early_stop_patience = getattr(args, 'early_stop_patience', 0)
        use_wandb = WANDB_AVAILABLE and not getattr(args, 'disable_wandb', False)
        
        self._log_section("TRAINING CONFIGURATION")
        print(f"  Dataset Type         : {self.dataset_type.upper()}")
        print(f"  Training Samples     : {train_samples if train_samples > 0 else 'ALL'}")
        print(f"  Validation Samples   : {val_samples if val_samples > 0 else 'ALL'}")
        print(f"  Evaluation Schedule  : {'Every ' + str(eval_steps) + ' steps' if eval_steps > 0 else 'Every epoch'}")
        print(f"  Logging Frequency    : Every {log_steps} steps")
        print(f"  Best Metric          : {self.best_metric}")
        print(f"  Backbone Training    : {'FROZEN' if freeze_backbone else 'TRAINABLE'}")
        print(f"  Wandb Integration    : {'ENABLED' if use_wandb else 'DISABLED'}")
        print(f"  Distributed Training : {'YES' if self.distributed else 'NO'}")
        print(f"  Early Stop Patience  : {early_stop_patience if early_stop_patience > 0 else 'DISABLED'} (measured in evaluations)")
        print()

    def _maybe_disable_slow_metrics(self):
        """Skip slow metrics (METEOR/SPICE) during training-only runs unless explicitly kept."""
        if getattr(self.args, 'keep_full_metrics', False):
            return
        if str(self.best_metric).upper() in {"METEOR", "SPICE"}:
            return
        slow_metrics = {'METEOR', 'SPICE'}
        paired = list(zip(list(cfg.SCORER.TYPES), list(cfg.SCORER.WEIGHTS)))
        filtered = [(m, w) for m, w in paired if m not in slow_metrics]
        removed = [m for m, _ in paired if m in slow_metrics]
        # Avoid empty list; fallback to original if everything would be stripped
        if filtered and len(filtered) != len(paired):
            cfg.SCORER.TYPES = [m for m, _ in filtered]
            cfg.SCORER.WEIGHTS = [w for _, w in filtered]
            if self.is_master:
                self._log(f"Disabled slow metrics for training: removed {removed}, keeping {cfg.SCORER.TYPES}", prefix="SCORER")

    def _clear_old_result_files(self):
        """清理旧的评估结果文件（仅在训练开始时执行）"""
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            return
            
        import glob
        old_result_files = glob.glob(os.path.join(result_folder, 'result_*.json'))
        if old_result_files:
            self._log_subsection("CLEARING OLD RESULT FILES")
            self._log(f"Found {len(old_result_files)} old result files", prefix="CLEANUP")
            for old_file in old_result_files:
                os.remove(old_file)
            self._log("Old result files cleared successfully", prefix="CLEANUP")
        else:
            self._log("No old result files to clear", prefix="CLEANUP")

    def _is_hf_model(self):
        model_type = str(getattr(cfg.MODEL, "TYPE", "")).lower()
        return (
            model_type.startswith("hf")
            or "blip" in model_type
            or "git" in model_type
            or "qwen" in model_type
            or "openrouter" in model_type
            or model_type.startswith("gpt")
            or "gpt" in model_type
        )

    def _is_hf_training(self):
        hf_cfg = getattr(cfg.MODEL, "HF", None)
        lora_enabled = bool(getattr(getattr(hf_cfg, "LORA", None), "ENABLED", False)) if hf_cfg else False
        trainable = bool(getattr(hf_cfg, "TRAINABLE", False)) if hf_cfg else False
        return self._is_hf_model() and (trainable or lora_enabled)

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(v) for v in obj)
        return obj

    def _unwrap_model(self):
        model = self.model
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return model.module
        return model

    def _save_lora_adapter(self, output_dir: str) -> bool:
        base_model = self._unwrap_model()
        if hasattr(base_model, "save_lora_adapter"):
            return base_model.save_lora_adapter(output_dir)
        return False

    def setup_wandb(self):
        """设置wandb日志记录"""
        self.use_wandb = WANDB_AVAILABLE and not getattr(self.args, 'disable_wandb', False)
        
        if not self.use_wandb:
            return
            
        # 只在主进程中初始化wandb
        if not self.is_master:
            return
            
        # 生成run name
        wandb_name = getattr(self.args, 'wandb_name', None)
        if wandb_name is None:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            wandb_name = f"{self.dataset_type}_{timestamp}"
            
        # wandb配置
        wandb_config = {
            'dataset': self.dataset_type,
            'train_samples': getattr(self.args, 'train_samples', 0),
            'val_samples': getattr(self.args, 'val_samples', 100),
            'eval_steps': getattr(self.args, 'eval_steps', 0),
            'log_steps': getattr(self.args, 'log_steps', 40),
            'freeze_backbone': getattr(self.args, 'freeze_backbone', False),
            'max_epoch': cfg.SOLVER.MAX_EPOCH,
            'batch_size': cfg.TRAIN.BATCH_SIZE,
            'learning_rate': cfg.SOLVER.BASE_LR,
            'model_type': cfg.MODEL.TYPE,
            'best_metric': self.best_metric,
        }
        
        # 初始化wandb
        try:
            wandb.init(
                project=getattr(self.args, 'wandb_project', 'ByteCaption'),
                name=wandb_name,
                config=wandb_config,
                resume="allow"
            )
            self._log(f"Wandb initialized: {wandb.run.url}", prefix="WANDB")
        except Exception as e:
            self._log(f"Failed to initialize wandb: {e}", level="WARNING", prefix="WANDB")
            self.use_wandb = False

    def setup_network(self):
        # 模型构建
        model = models.create(cfg.MODEL.TYPE)
        is_hf = self._is_hf_model()
        
        load_weights = getattr(self.args, 'load_weights', False)
        if load_weights and not is_hf:
            self._log("Loading pretrained weights to backbone...", prefix="MODEL")
            weights = torch.load("byteformer_hf_migration/weights/imagenet_jpeg_q60_k4_w128.pt", map_location='cpu')
            # 加载backbone部分权重
            model_state = model.backbone.byteformer.state_dict()
            pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
            model.backbone.byteformer.load_state_dict(pretrained_state, strict=False)
        
        # 根据启动参数决定是否冻结backbone
        freeze_backbone = getattr(self.args, 'freeze_backbone', False)
        if not is_hf:
            for _name, _weight in model.backbone.named_parameters():
                _weight.requires_grad = not freeze_backbone
            
        if self.is_master:
            self._log("Model architecture loaded successfully", prefix="MODEL")

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        else:
            if torch.cuda.is_available():
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model = model.to(self.device)

        # 如果resume > 0，则需要导入参数
        # 此处导入参数到CPU上？
        if not is_hf:
            if self.args.resume > 0:
                self.model.load_state_dict(
                    torch.load(self.snapshot_path("caption_model", self.args.resume),
                        map_location=lambda storage, loc: storage)
                )
            elif self.args.resume == -1:
                self.model.load_state_dict(
                    torch.load("/root/autodl-tmp/ByteCaption/PureT/experiments/ByteCaption_XE/byteformer_20k/best_model.pth",
                    map_location=lambda storage, loc: storage)
                )

        # 判断是否导入epoch
        self.load_epoch = -1
        self.load_iteration = -1
        if self.args.load_epoch:
            self.load_epoch = self.args.resume - 1  # 保存的resume名称从1计数
            # 113287是训练样本数量
            self.load_iteration = int(self.args.resume * 113287 / cfg.TRAIN.BATCH_SIZE)

        # 训练优化器
        # load_iteration为scheduler中使用的last_epoch，
        # 用于简单粗略的恢复学习率，只对NoamOpt作用
        # 完整恢复optimizer，还是得保存checkpoint文件
        self.optim = Optimizer(self.model, self.load_iteration, is_master=self.is_master)
        # 训练损失计算
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).to(self.device)
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).to(self.device)

    # 模型验证
    def eval(self, epoch):
        # Reset latest validation metrics before a new evaluation run
        self.latest_val_res = None
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        val_res = None
        test_res = None
        if self.val_evaler is not None:
            val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
            self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
            self.logger.info(str(val_res))
            self.latest_val_res = val_res
            
            # Wandb记录验证指标
            if self.use_wandb and val_res is not None:
                wandb_log = {'val/epoch': epoch + 1}
                for metric, value in val_res.items():
                    wandb_log[f'val/{metric}'] = value
                wandb.log(wandb_log)
        else:
            self.logger.info('VAL evaluation skipped (no COCO files).')

        if self.test_evaler is not None:
            # 只在最后几个epoch或特定间隔评估测试集
            if (epoch + 1) >= cfg.SOLVER.MAX_EPOCH - 2 or (epoch + 1) % 5 == 0:
                test_res = self.test_evaler(self.model,'test_' + str(epoch + 1))
                self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
                self.logger.info(str(test_res))
                
                # Wandb记录测试指标
                if self.use_wandb and test_res is not None:
                    wandb_log = {'test/epoch': epoch + 1}
                    for metric, value in test_res.items():
                        wandb_log[f'test/{metric}'] = value
                    wandb.log(wandb_log)
            else:
                self.logger.info('TEST evaluation skipped for this epoch.')
        else:
            self.logger.info('TEST evaluation skipped (no test evaler).')

        if val_res is None:
            return None
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    # 基于步数的评估
    def eval_by_steps(self, iteration, epoch=None, step_in_epoch=None):
        if self.distributed and dist.get_rank() > 0:
            return None

        val_res = None
        if self.val_evaler is not None:
            val_res = self.val_evaler(self.model, f'step_{iteration}')
            self.latest_val_res = val_res
            
            # Wandb记录步数评估指标
            if self.use_wandb and val_res is not None:
                wandb_log = {'val_step/iteration': iteration}
                
                # 添加epoch进度信息
                if epoch is not None and step_in_epoch is not None:
                    epoch_progress = step_in_epoch / len(self.training_loader)
                    wandb_log['val_step/epoch'] = epoch + epoch_progress
                
                for metric, value in val_res.items():
                    wandb_log[f'val_step/{metric}'] = value
                wandb.log(wandb_log, step=iteration)
        else:
            # 使用日志而不是print来避免与进度条冲突
            self.logger.info('VAL evaluation skipped (no evaler).')

        if val_res is None:
            return None
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        # 返回模型路径：experiments/snapshot/{MODELNAME}_{epoch}.pth
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    # 保存模型
    def save_model(self, epoch=None, val_score=None, is_step_eval=False, iteration=None):
        # if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
        #     return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        improved = False
        hf_cfg = getattr(cfg.MODEL, "HF", None)
        lora_cfg = getattr(hf_cfg, "LORA", None) if hf_cfg else None
        lora_enabled = bool(getattr(lora_cfg, "ENABLED", False)) if lora_cfg else False
        save_full_model = bool(getattr(lora_cfg, "SAVE_FULL_MODEL", True)) if lora_cfg else True
        use_lora_adapter = self._is_hf_model() and lora_enabled and not save_full_model
        
        # 保存当前模型（仅在epoch快照时）
        save_snapshot = False
        if (not is_step_eval) and (epoch is not None):
            is_snapshot_iter = ((epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0) if cfg.SOLVER.SNAPSHOT_ITERS > 0 else False
            is_last_epoch = (epoch + 1) == cfg.SOLVER.MAX_EPOCH
            save_snapshot = is_snapshot_iter or is_last_epoch
            if save_snapshot:
                if use_lora_adapter:
                    self._log("Skipping full checkpoint snapshot for LoRA model.", prefix="CHECKPOINT")
                else:
                    current_model_path = self.snapshot_path("caption_model", epoch + 1)
                    torch.save(self.model.state_dict(), current_model_path)
                    self._log(f"Saving snapshot to: {current_model_path}", prefix="CHECKPOINT")

        # 基于指标保存最佳模型（支持step或epoch评估）
        if val_score is not None and val_score > self.best_cider:
            self.best_cider = val_score
            if epoch is not None:
                self.best_epoch = epoch + 1
            if iteration is not None:
                self.best_step = iteration
            if use_lora_adapter:
                best_path = os.path.join(snapshot_folder, "best_lora")
                saved = self._save_lora_adapter(best_path)
                if not saved:
                    self._log("LoRA adapter save failed; falling back to full checkpoint.", level="WARNING", prefix="CHECKPOINT")
                    best_path = os.path.join(snapshot_folder, "best_model.pth")
                    torch.save(self.model.state_dict(), best_path)
            else:
                best_path = os.path.join(snapshot_folder, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)
            improved = True
            if self.is_master:
                location = f"step {iteration}" if is_step_eval and iteration is not None else f"epoch {epoch + 1}" if epoch is not None else "unknown"
                self._log(f"Best model updated at {location}: score={val_score:.4f}", prefix="CHECKPOINT")
        
        # 检查并清理旧的checkpoint文件
        if cfg.SOLVER.MAX_CHECKPOINTS > 0:
            self._cleanup_old_checkpoints(snapshot_folder)
        
        return improved
    
    def _cleanup_old_checkpoints(self, snapshot_folder):
        """清理旧的checkpoint文件，只保留最新的MAX_CHECKPOINTS个"""
        import glob
        import os
        
        # 查找所有caption_model的checkpoint文件
        pattern = os.path.join(snapshot_folder, "caption_model_*.pth")
        checkpoint_files = glob.glob(pattern)
        
        # 按修改时间排序，最新的在最后
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
        
        # 如果文件数量超过限制，删除最旧的文件
        while len(checkpoint_files) > cfg.SOLVER.MAX_CHECKPOINTS:
            oldest_file = checkpoint_files.pop(0)
            try:
                os.remove(oldest_file)
                self._log(f"Removed old checkpoint: {oldest_file}", prefix="CHECKPOINT")
            except OSError as e:
                self._log(f"Failed to remove {oldest_file}: {e}", level="ERROR", prefix="CHECKPOINT")

    def _update_patience(self, improved, logger=None, context_label=""):
        """基于评估结果更新早停计数，单位为评估触发次数"""
        if self.early_stop_patience <= 0 or improved is None:
            return False

        if improved:
            self.evals_since_improvement = 0
            return False

        self.evals_since_improvement += 1
        if self.is_master and logger is not None:
            logger(
                f"No {self.best_metric} improvement for {self.evals_since_improvement}/{self.early_stop_patience} evaluations "
                f"(best {self.best_cider:.4f} at {self._best_marker_label()})."
            )

        if self.evals_since_improvement >= self.early_stop_patience:
            if self.is_master and logger is not None:
                logger(
                    f"Early stopping after {self.early_stop_patience} evaluations (last check: {context_label})."
                )
            return True

        return False

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
        device = input_seq.device
        seq_mask = (input_seq > 0).long().to(device)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs

    # 返回scheduled sampling概率
    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    # 训练数据显示
    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    # 模型损失计算过程
    def forward(self, kwargs):
        if self.rl_stage == False:
            # XE训练过程损失计算
            logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
        else:
            """
            # SCST训练过程损失计算 -- 参考M2Transformer
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
            
            kwargs['BEAM_SIZE'] = 5
            kwargs['OUT_SIZE'] = 5
            
            # 前向计算，beam search结果
            seq_beam, logP_beam = self.model.module.decode_beam(**kwargs)
            
            mask = seq_beam > 0  # [10, 5, 17]
            mask = mask.view(-1, mask.size()[-1])  # [50, 17]
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            mask = mask.view(-1, kwargs['BEAM_SIZE'], mask.size()[-1]) # [10, 5, 17]
            
            # print(seq_beam[0])
            # print(logP_beam[0])
            # 计算Beam Search的结果与Ground Truth的CIDEr Reward，估算损失
            ids = utils.expand_numpy(ids, kwargs['BEAM_SIZE'])
            seq_beam = seq_beam.view(-1, seq_beam.size()[-1])
            rewards_beam, rewards_info_beam = self.scorer(ids, seq_beam.data.cpu().numpy().tolist())
            # print('rewards:', rewards_beam.mean(), rewards_beam.min(), rewards_beam.max())
            '''
            self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)
            sents = utils.decode_sequence(self.vocab, seq_beam.data)
            # Debug: show first 5 generated sentences (commented out for cleaner output)
            # self._log(f"Sample sentences: {sents[:5]}", prefix="DEBUG")
            '''
            
            rewards_beam = torch.from_numpy(rewards_beam).to(self.device).view(att_feats.size()[0], kwargs['BEAM_SIZE'])
            reward_baseline = torch.mean(rewards_beam, -1, keepdim=True)
            # loss = -torch.mean(logP_beam, -1) * (rewards_beam - reward_baseline)
            loss = -(torch.sum(logP_beam * mask, -1) / torch.sum(mask, -1)) * (rewards_beam - reward_baseline)
            loss = loss.mean()
            
            loss_info = {}
            loss_info['reward_baseline'] = reward_baseline.mean().item()
            return loss, loss_info
            
            """
            # """
            # SCST训练过程损失计算 -- 参考ruotian luo的new scst
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
            
            # 不使用beam search，采样，将输入数据进行扩充
            ids = utils.expand_numpy(ids)
            gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
            
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            # 采样（采样函数与ruotian luo不一樣，來源於XLAN Net）
            seq_sample, logP_sample = self.model.module.decode(**kwargs)
            # 计算Sample生成的句子与GTs之间的CIDEr得分
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())
            
            mask = seq_sample > 0
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            rewards_sample = torch.from_numpy(rewards_sample).to(self.device).view(-1, 5)
            reward_baseline = (torch.sum(rewards_sample, 1, keepdim=True) - rewards_sample) / (rewards_sample.size()[1] - 1)
            
            loss = - logP_sample * mask * (rewards_sample - reward_baseline).view(-1, 1)
            loss = torch.sum(loss) / torch.sum(mask)
            
            loss_info = {}
            loss_info['reward_baseline'] = reward_baseline.mean().item()
            return loss, loss_info
            # """
            
            """
            # SCST训练过程损失计算 -- 初始的SCST，使用Greedy结果作为baseline
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

            ##############
            # Greedy
            ##############
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.module.decode(**kwargs)
            self.model.train()
            # 计算greedy生成的句子与GTs之间的CIDEr得分
            rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
            rewards_max = utils.expand_numpy(rewards_max)
            
            ids = utils.expand_numpy(ids)
            gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)

            ##############
            # Sample
            ##############
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            seq_sample, logP_sample = self.model.module.decode(**kwargs)
            # 计算Sample生成的句子与GTs之间的CIDEr得分
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())

            # print('rewards_sample:', rewards_sample)
            # print('rewards_max:', rewards_max)
            # 计算sample句子和greedy句子得分差值
            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().to(self.device)
            # 估算损失
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)

            loss_info = {}
            for key in rewards_info_sample:
                # loss_info[key + '_sample'] = rewards_info_sample[key]
                loss_info['reward'] = rewards_info_sample[key]
            for key in rewards_info_max:
                # loss_info[key + '_max'] = rewards_info_max[key]
                loss_info['reward_baseline'] = rewards_info_max[key]
            # """

        return loss, loss_info

    # 模型训练过程
    def train(self):
        self.model.train()
        # self.optim.zero_grad()

        iteration = self.load_iteration + 1
        eval_steps = getattr(self.args, 'eval_steps', 0)
        log_steps = getattr(self.args, 'log_steps', 20)
        
        # 清理旧的评估结果文件（仅在训练开始时执行一次）
        if self.is_master:
            self._clear_old_result_files()
        
        # 计算总的训练步数
        total_epochs = cfg.SOLVER.MAX_EPOCH - (self.load_epoch + 1)
        if total_epochs > 0:
            # 先设置一个epoch的loader来获取每个epoch的步数
            self.setup_loader(self.load_epoch + 1)
            steps_per_epoch = len(self.training_loader)
            total_steps = total_epochs * steps_per_epoch
            
            # 更新optimizer的训练参数，以便自动计算学习率调度器的迭代数
            self.optim.update_training_params(total_epochs, steps_per_epoch)
            
            if self.is_master:
                self._log_section("TRAINING PLAN")
                print(f"  Total Epochs    : {total_epochs}")
                print(f"  Steps per Epoch : {steps_per_epoch:,}")
                print(f"  Total Steps     : {total_steps:,}")
                print(f"  Learning Rate   : {self.optim.get_lr()}")
                print()
        else:
            total_steps = 0
        
        # 创建跨越所有epoch的总进度条 (仅主进程显示)
        pbar_ctx = tqdm.tqdm(
            desc='Overall Training Progress',
            unit='step',
            total=total_steps,
            disable=not self.is_master
        )
        with pbar_ctx as overall_pbar:
            stop_training = False
            # Epoch迭代
            for epoch in range(self.load_epoch + 1, cfg.SOLVER.MAX_EPOCH):
                if stop_training:
                    break
                if self.is_master:
                    overall_pbar.write(f"Current learning rate: {self.optim.get_lr()}")
                if epoch >= cfg.TRAIN.REINFORCEMENT.START:
                    self.rl_stage = True
                # 设置DataLoader
                self.setup_loader(epoch)

                # start = time.time()
                # 自动求均值
                # data_time = AverageMeter()
                # batch_time = AverageMeter()
                # losses = AverageMeter()
                
                running_loss = .0
                running_reward_baseline = .0
                loss_window = deque(maxlen=log_steps if log_steps > 0 else None)
                reward_baseline_window = deque(maxlen=log_steps if log_steps > 0 else None)
                self.optim.zero_grad()
                # 每一个Epoch内部Iteration迭代 - 不再使用epoch进度条，只更新总进度条
                for step_idx, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask) in enumerate(self.training_loader):
                    if stop_training:
                        break
                    # data_time.update(time.time() - start)
                    input_seq = self._move_to_device(input_seq)
                    target_seq = self._move_to_device(target_seq)
                    gv_feat = self._move_to_device(gv_feat)
                    att_feats = self._move_to_device(att_feats)
                    if att_mask is not None:
                        att_mask = self._move_to_device(att_mask)

                    kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
                    # 1、计算模型损失（XE训练 或 SCST训练）
                    loss, loss_info = self.forward(kwargs)
                    # 2、梯度累积与反向传播
                    loss_value = loss.item()
                    scaled_loss = loss / self.grad_accum_steps
                    needs_sync = (
                        self.distributed
                        and self.grad_accum_steps > 1
                        and ((step_idx + 1) % self.grad_accum_steps != 0)
                    )
                    if needs_sync:
                        with self.model.no_sync():
                            scaled_loss.backward()
                    else:
                        scaled_loss.backward()

                    update_now = ((step_idx + 1) % self.grad_accum_steps == 0) or (
                        step_idx + 1 == len(self.training_loader)
                    )
                    if update_now:
                        utils.clip_gradient(
                            self.optim.optimizer, self.model,
                            cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP
                        )
                        self.optim.step()
                        self.optim.scheduler_step('Iter')
                        self.optim.zero_grad()

                    # batch_time.update(time.time() - start)
                    # start = time.time()
                    # losses.update(loss.item())
                    # self.display(iteration, data_time, batch_time, losses, loss_info)
                    # tqdm 迭代信息更新
                    running_loss += loss_value
                    loss_window.append(loss_value)
                    recent_avg_loss = sum(loss_window) / len(loss_window)
                    epoch_avg_loss = running_loss / (step_idx + 1)

                    recent_reward_baseline = None
                    if self.rl_stage and 'reward_baseline' in loss_info:
                        reward_baseline_value = loss_info['reward_baseline']
                        running_reward_baseline += reward_baseline_value
                        reward_baseline_window.append(reward_baseline_value)
                        recent_reward_baseline = sum(reward_baseline_window) / len(reward_baseline_window)
                    
                    # 添加训练调试信息和wandb记录
                    if self.is_master and log_steps > 0 and iteration % log_steps == 0:  # 根据设定的log_steps输出详细信息
                        current_lr_list = self.optim.get_lr()
                        current_lr = current_lr_list[0] if current_lr_list else 0
                        overall_pbar.write(
                            f"Step {iteration}: loss={loss_value:.4f}, avg_loss_last_{len(loss_window)}={recent_avg_loss:.4f}, "
                            f"epoch_avg={epoch_avg_loss:.4f}, lr={current_lr:.6f}"
                        )
                        
                        # Wandb记录训练损失
                        if self.use_wandb:
                            # 计算epoch进度
                            epoch_progress = (step_idx + 1) / len(self.training_loader)
                            log_dict = {
                                'train/loss': loss_value,
                                'train/avg_loss_window': recent_avg_loss,
                                'train/avg_loss_epoch': epoch_avg_loss,
                                'train/learning_rate': current_lr,
                                'train/epoch': epoch + epoch_progress,
                            }
                            
                            # 如果是SCST阶段，记录额外信息
                            if self.rl_stage and 'reward_baseline' in loss_info:
                                log_dict['train/reward_baseline'] = loss_info['reward_baseline']
                                if recent_reward_baseline is not None:
                                    log_dict['train/reward_baseline_window'] = recent_reward_baseline
                                
                            wandb.log(log_dict, step=iteration)
                        
                    # 更新总进度条，包含更详细的信息
                    if self.is_master:
                        if not self.rl_stage:
                            lr_list = self.optim.get_lr()
                            overall_pbar.set_postfix(
                                epoch=epoch,
                                loss='%.3f' % recent_avg_loss,
                                lr='%.2e' % (lr_list[0] if lr_list else 0)
                            )
                        else:
                            lr_list = self.optim.get_lr()
                            rb_display = recent_reward_baseline if recent_reward_baseline is not None else (running_reward_baseline / (step_idx + 1))
                            overall_pbar.set_postfix({
                                'epoch': epoch,
                                'loss/r_b': '%.3f/%.3f' % (recent_avg_loss, rb_display),
                                'lr': '%.2e' % (lr_list[0] if lr_list else 0)
                            })
                        overall_pbar.update()
                    
                    # print(str(self.optim.get_lr()))
                    iteration += 1

                    # 基于步数的评估
                    if eval_steps > 0 and iteration % eval_steps == 0:
                        if self.is_master:
                            overall_pbar.write(f"{'='*60}")
                            overall_pbar.write(f"Step-based Evaluation at iteration {iteration}")
                            overall_pbar.write(f"Average training loss: {epoch_avg_loss:.4f}")
                            overall_pbar.write(f"{'='*60}")
                        val = self.eval_by_steps(iteration, epoch, step_idx + 1)
                        # 可选：基于验证结果进行学习率调度
                        # self.optim.scheduler_step('Epoch', val)

                        metric_score = None
                        if self.latest_val_res is not None:
                            metric_score = self.latest_val_res.get(self.best_metric)

                        improved = None
                        if metric_score is not None:
                            improved = self.save_model(epoch, metric_score, is_step_eval=True, iteration=iteration) or False

                        stop_trigger = self._update_patience(improved, overall_pbar.write if self.is_master else None, context_label=f"step {iteration}")

                        if self.distributed:
                            stop_tensor = torch.tensor(1 if stop_trigger else 0, device=self.device)
                            dist.broadcast(stop_tensor, src=0)
                            stop_trigger = stop_tensor.item() == 1

                        if stop_trigger:
                            stop_training = True
                            break

                    if self.distributed:
                        dist.barrier()
            
                if stop_training:
                    break

                val = self.eval(epoch)
                print("一轮结束后的val", val)

                metric_score = None
                if self.latest_val_res is not None:
                    metric_score = self.latest_val_res.get(self.best_metric)

                # 每一个Epoch结束保存模型（基于 best_metric 最佳）
                improved = None
                if metric_score is not None:
                    improved = self.save_model(epoch, metric_score) or False
                # 模型验证测试，返回的val仅用于SCST训练过程
                # 如果使用基于步数的评估，可以跳过epoch评估或减少频率
                # if eval_steps == 0:  # 只有在不使用步数评估时才进行epoch评估
                #     val = self.eval(epoch)
                # else:
                #     val = None
                #     if self.is_master:
                #         overall_pbar.write(f"Epoch {epoch + 1} completed. Using step-based evaluation, skipping epoch evaluation.")
                    
                # 4（SCST）、优化器lr更新（用于SCST训练），在XE训练时不起作用
                # 4 (XE)、优化器lr更新，当使用Step学习率策略时作用
                self.optim.scheduler_step('Epoch', val)
                self.scheduled_sampling(epoch)
                
                # 早停检查（基于验证集 best_metric）
                stop_trigger = self._update_patience(improved, overall_pbar.write if self.is_master else None, context_label=f"epoch {epoch + 1}")

                if self.distributed:
                    stop_tensor = torch.tensor(1 if stop_trigger else 0, device=self.device)
                    dist.broadcast(stop_tensor, src=0)
                    stop_trigger = stop_tensor.item() == 1

                if stop_trigger:
                    stop_training = True
                    break
                
                if self.distributed:
                    dist.barrier()
        
        # 训练完成后关闭wandb
        if self.use_wandb and self.is_master:
            wandb.finish()

def parse_args():
    
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-2)
    parser.add_argument("--load_epoch", action='store_true')
    parser.add_argument("--dataset", type=str, default='coco', choices=['coco', 'flickr8k'],
                        help='Choose dataset: coco or flickr8k (default: coco)')
    parser.add_argument("--train_samples", type=int, default=0,
                        help='Number of training samples to use (0 for all)')
    parser.add_argument("--val_samples", type=int, default=50,
                        help='Number of validation samples to use (0 for all)')
    parser.add_argument("--eval_steps", type=int, default=50,
                        help='Evaluate every N steps (0 for every epoch)')
    parser.add_argument("--log_steps", type=int, default=40,
                        help='Print detailed training log every N steps')
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help='Early stop training if CIDEr does not improve for this many evaluations (0 disables)')
    parser.add_argument("--freeze_backbone", action='store_true',
                        help='Freeze backbone parameters during training')
    parser.add_argument("--load_weights", action='store_true',
                        help='Path to pretrained weights file (.pth) to load')
    parser.add_argument("--keep_full_metrics", action='store_true',
                        help='Do not strip slow metrics (METEOR/SPICE) during training evaluations')
    parser.add_argument("--best_metric", type=str, default="CIDEr",
                        help="Metric name to select best checkpoint (e.g., SPICE, CIDEr)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")
    parser.add_argument("--wandb_project", type=str, default="ByteCaption",
                        help='Wandb project name')
    parser.add_argument("--wandb_name", type=str, default=None,
                        help='Wandb run name (auto-generated if not specified)')
    parser.add_argument("--disable_wandb", action='store_true',
                        help='Disable wandb logging')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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

if __name__ == '__main__':
    args = parse_args()
    print('[SYSTEM] Called with args:')
    print(args)

    if args.folder is not None:
        # 根据数据集选择配置文件
        dataset_type = getattr(args, 'dataset', 'coco').lower()
        if dataset_type == 'coco':
            config_file = 'config_coco.yml'
        elif dataset_type == 'flickr8k':
            config_file = 'config_flickr8k.yml'
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        config_path = os.path.join(args.folder, config_file)
        if os.path.exists(config_path):
            cfg_from_file(config_path)
            print(f"[CONFIG] Loaded config: {config_path}")
        else:
            print(f"[WARNING] Config file not found: {config_path}")
            if dataset_type == 'coco':
                print("[CONFIG] Falling back to COCO config")
                cfg_from_file(os.path.join(args.folder, 'config_coco.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()
