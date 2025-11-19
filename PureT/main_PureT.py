import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models

# COCO and Flickr8k components
from datasets_.coco_dataset_hf import CocoDataset
from datasets_.flickr8k_dataset_hf import Flickr8kDataset
from datasets_.data_loader import load_train
from evaluation.evaler_coco import CocoEvaler
from evaluation.evaler_flickr8k import Flickr8kEvaler
from scorer.coco_scorer import CocoScorer
from scorer.flickr8k_scorer import Flickr8kScorer

import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

"""
cd /d/MLLMs/corenet && python PureT/main.py --folder ./PureT/experiments_PureT/PureT_XE --eval_steps 50
cd /root/autodl-tmp/corenet && PYTHONPATH=/root/autodl-tmp/corenet python PureT/main.py --folder ./PureT/experiments_PureT/PureT_XE --eval_steps 300
cd /root/autodl-tmp/corenet && PYTHONPATH=/root/autodl-tmp/corenet torchrun --nproc_per_node=2 --master_port=12355 PureT/main.py --folder ./PureT/experiments_PureT/PureT_XE --eval_steps 300 --enable_eval_loss --dataset coco
"""

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        
        # 数据集类型 (must be defined before setup_dataset / setup_evaler)
        self.dataset_type = getattr(args, 'dataset', 'flickr8k').lower()
        print(f"Dataset type: {self.dataset_type}")
        
        # 显示数据集配置
        train_samples = getattr(args, 'train_samples', 200)
        val_samples = getattr(args, 'val_samples', 100)
        eval_steps = getattr(args, 'eval_steps', 0)
        log_steps = getattr(args, 'log_steps', 20)
        enable_eval_loss = getattr(args, 'enable_eval_loss', False)
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Training samples: {train_samples if train_samples > 0 else 'ALL'}")
        print(f"Validation samples: {val_samples if val_samples > 0 else 'ALL'}")
        print(f"Evaluation: {'Every ' + str(eval_steps) + ' steps' if eval_steps > 0 else 'Every epoch'}")
        print(f"Log steps: Every {log_steps} steps")
        print(f"Eval XE Loss: {'ENABLED' if enable_eval_loss else 'DISABLED'}")
        print("="*60)
        
        # 设置随机数种子
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        # 单机多卡
        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        
        if self.distributed:
            self.local_rank = init_distributed_mode()
        else:
            self.local_rank = 0
            
        # Choose device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SCST标记
        self.rl_stage = False
        # 设置日志写入
        self.setup_logging()
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
            
    def setup_evaler(self):
        # 使用命令行参数或默认值
        val_samples = getattr(self.args, 'val_samples', 100)
        if val_samples == 0:
            val_samples = None  # None表示使用所有样本

        # 获取enable_eval_loss参数
        enable_eval_loss = getattr(self.args, 'enable_eval_loss', False)

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
            print(f"Validation dataset (COCO): Using {val_samples if val_samples else 'ALL'} samples")
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
            print(f"Validation dataset (Flickr8k): Using {val_samples if val_samples else 'ALL'} samples")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        print(f"XE Loss during evaluation: {'ENABLED' if enable_eval_loss else 'DISABLED'}")
        
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
                max_samples=train_samples
            )
            print(f"Training dataset (COCO): Using {train_samples if train_samples else 'ALL'} samples")
        elif self.dataset_type == 'flickr8k':
            self.training_dataset = Flickr8kDataset(
                image_ids_path=train_id_path,
                input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
                target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
                gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
                seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
                max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
                max_samples=train_samples
            )
            print(f"Training dataset (Flickr8k): Using {train_samples if train_samples else 'ALL'} samples")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # 为了向后兼容，保留coco_set引用
        self.coco_set = self.training_dataset

    # DataLoader
    def setup_loader(self, epoch):
        self.training_loader = load_train(
            self.distributed, epoch, self.training_dataset)

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

    def setup_network(self):
        # 模型构建
        model = models.create(cfg.MODEL.TYPE)
        print(model)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )
        else:
            if torch.cuda.is_available():
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model = model.to(self.device)

        # 如果resume > 0，则需要导入参数
        # 此处导入参数到CPU上？
        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
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
        self.optim = Optimizer(self.model, self.load_iteration)
        # 训练损失计算
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).to(self.device)
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).to(self.device)

    # 模型验证
    def eval(self, epoch):
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
        else:
            self.logger.info('VAL evaluation skipped (no COCO files).')

        if self.test_evaler is not None:
            # 只在最后几个epoch或特定间隔评估测试集
            if (epoch + 1) >= cfg.SOLVER.MAX_EPOCH - 2 or (epoch + 1) % 5 == 0:
                test_res = self.test_evaler(self.model,'test_' + str(epoch + 1))
                self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
                self.logger.info(str(test_res))
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
    def eval_by_steps(self, iteration):
        if self.distributed and dist.get_rank() > 0:
            return None

        val_res = None
        if self.val_evaler is not None:
            val_res = self.val_evaler(self.model, f'step_{iteration}')
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
    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        
        # 保存当前模型
        current_model_path = self.snapshot_path("caption_model", epoch+1)
        torch.save(self.model.state_dict(), current_model_path)
        
        # 检查并清理旧的checkpoint文件
        if cfg.SOLVER.MAX_CHECKPOINTS > 0:
            self._cleanup_old_checkpoints(snapshot_folder)
    
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
                print(f"Removed old checkpoint: {oldest_file}")
            except OSError as e:
                print(f"Failed to remove {oldest_file}: {e}")

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
            print(sents[:5])
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
        
        # 计算总的训练步数
        total_epochs = cfg.SOLVER.MAX_EPOCH - (self.load_epoch + 1)
        if total_epochs > 0:
            # 先设置一个epoch的loader来获取每个epoch的步数
            self.setup_loader(self.load_epoch + 1)
            steps_per_epoch = len(self.training_loader)
            total_steps = total_epochs * steps_per_epoch
            
            print(f"Training plan: {total_epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps")
        else:
            total_steps = 0
        
        # 创建跨越所有epoch的总进度条
        with tqdm.tqdm(desc='Overall Training Progress', unit='step', total=total_steps) as overall_pbar:
            # Epoch迭代
            for epoch in range(self.load_epoch + 1, cfg.SOLVER.MAX_EPOCH):
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
                # 每一个Epoch内部Iteration迭代 - 不再使用epoch进度条，只更新总进度条
                for _, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask) in enumerate(self.training_loader):
                    # data_time.update(time.time() - start)
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    gv_feat = gv_feat.to(self.device)
                    att_feats = att_feats.to(self.device)
                    att_mask = att_mask.to(self.device)

                    kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
                    # 1、计算模型损失（XE训练 或 SCST训练）
                    loss, loss_info = self.forward(kwargs)
                    # 2、梯度清零（清空过往梯度）
                    self.optim.zero_grad()
                    # 3、计算新梯度及梯度裁剪
                    loss.backward()  # 非混合精度训练
                    utils.clip_gradient(self.optim.optimizer, self.model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    # 4、权重更新
                    self.optim.step() # 非混合精度训练
                    # 5、（XE）、优化器lr更新（用于XE训练），在SCST时不起作用
                    self.optim.scheduler_step('Iter')

                    # batch_time.update(time.time() - start)
                    # start = time.time()
                    # losses.update(loss.item())
                    # self.display(iteration, data_time, batch_time, losses, loss_info)
                    # tqdm 迭代信息更新
                    running_loss += loss.item()
                    
                    # 添加训练调试信息
                    if iteration % log_steps == 0:  # 根据设定的log_steps输出详细信息
                        current_lr = self.optim.get_lr()[0] if self.optim.get_lr() else 0
                        # 使用 tqdm.write 避免与进度条重叠
                        overall_pbar.write(f"Step {iteration}: loss={loss.item():.4f}, avg_loss={running_loss / (_ + 1):.4f}, lr={current_lr:.6f}")
                        
                    # 更新总进度条，包含更详细的信息
                    if not self.rl_stage:
                        overall_pbar.set_postfix(
                            epoch=epoch,
                            loss='%.3f' % (running_loss / (_ + 1)),
                            lr='%.2e' % (self.optim.get_lr()[0] if self.optim.get_lr() else 0)
                        )
                    else:
                        running_reward_baseline += loss_info['reward_baseline']
                        overall_pbar.set_postfix({
                            'epoch': epoch,
                            'loss/r_b': '%.3f/%.3f' % (running_loss / (_ + 1), running_reward_baseline / (_ + 1)),
                            'lr': '%.2e' % (self.optim.get_lr()[0] if self.optim.get_lr() else 0)
                        })
                    overall_pbar.update()
                    
                    # print(str(self.optim.get_lr()))
                    iteration += 1

                    # 基于步数的评估
                    if eval_steps > 0 and iteration % eval_steps == 0:
                        overall_pbar.write(f"{'='*60}")
                        overall_pbar.write(f"Step-based Evaluation at iteration {iteration}")
                        overall_pbar.write(f"Average training loss: {running_loss / _:.4f}")
                        overall_pbar.write(f"{'='*60}")
                        val = self.eval_by_steps(iteration)
                        # 可选：基于验证结果进行学习率调度
                        # self.optim.scheduler_step('Epoch', val)

                    if self.distributed:
                        dist.barrier()
            
            # 每一个Epoch结束保存模型
            self.save_model(epoch)
            # 模型验证测试，返回的val仅用于SCST训练过程
            # 如果使用基于步数的评估，可以跳过epoch评估或减少频率
            if eval_steps == 0:  # 只有在不使用步数评估时才进行epoch评估
                val = self.eval(epoch)
            else:
                val = None
                overall_pbar.write(f"Epoch {epoch + 1} completed. Using step-based evaluation, skipping epoch evaluation.")
                
            # 4（SCST）、优化器lr更新（用于SCST训练），在XE训练时不起作用
            # 4 (XE)、优化器lr更新，当使用Step学习率策略时作用
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)
            
            if self.distributed:
                dist.barrier()

def parse_args():
    
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument("--load_epoch", action='store_true')
    parser.add_argument("--train_samples", type=int, default=0,
                        help='Number of training samples to use (0 for all)')
    parser.add_argument("--val_samples", type=int, default=50,
                        help='Number of validation samples to use (0 for all)')
    parser.add_argument("--eval_steps", type=int, default=50,
                        help='Evaluate every N steps (0 for every epoch)')
    parser.add_argument("--log_steps", type=int, default=20,
                        help='Print detailed training log every N steps')
    parser.add_argument("--enable_eval_loss", action='store_true',
                        help='Enable XE loss calculation during evaluation (may cause memory issues)')
    parser.add_argument("--dataset", type=str, default="flickr8k",
                        help='Dataset type: coco or flickr8k')

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
        print('No distributed environment detected, using single GPU/CPU')
        return 0
    
    torch.cuda.set_device(local_gpu)
    print('| distributed init (rank {}): env://'.format(local_rank), flush=True)
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
    print('Called with args:')
    print(args)

    if args.folder is not None:
        if args.dataset == 'coco':
            cfg_from_file(os.path.join(args.folder, 'config_pureT_coco.yml'))
        elif args.dataset == 'flickr8k':
            cfg_from_file(os.path.join(args.folder, 'config_flickr8k.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()
