import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.config import cfg
import lr_scheduler
from optimizer.radam import RAdam, AdamW

class Optimizer(nn.Module):
    def __init__(self, model, begin_iteration, total_epochs=None, steps_per_epoch=None, is_master=True):
        super(Optimizer, self).__init__()
        self.last_epoch = begin_iteration
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.is_master = is_master
        self.setup_optimizer(model)
    
    def update_training_params(self, total_epochs, steps_per_epoch):
        """
        更新训练参数，用于自动计算迭代次数
        Args:
            total_epochs (int): 总训练轮数
            steps_per_epoch (int): 每轮步数
        """
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        
        # 如果使用的是需要总迭代数的调度器，重新创建它们
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            lr_type = cfg.SOLVER.LR_POLICY.TYPE
            if lr_type in ['Linear', 'CosineAnnealing', 'CosineWarmRestarts']:
                if self.is_master:
                    total_iters = self._calculate_total_iters()
                    print(f"Updating LR scheduler with total_iters: {total_iters}")
                # 重新设置调度器
                self._setup_scheduler()
    
    def _calculate_total_iters(self):
        """
        根据实际设置的训练轮数和数据量自动计算最大迭代数
        Returns:
            int: 总迭代数
        """
        # 如果配置文件中明确设置了TOTAL_ITERS，则优先使用
        if hasattr(cfg.SOLVER.LR_POLICY, 'TOTAL_ITERS') and cfg.SOLVER.LR_POLICY.TOTAL_ITERS is not None:
            return cfg.SOLVER.LR_POLICY.TOTAL_ITERS
        
        # 如果提供了总轮数和每轮步数，则自动计算
        if self.total_epochs is not None and self.steps_per_epoch is not None:
            return self.total_epochs * self.steps_per_epoch
        
        # 否则使用MAX_EPOCH和steps_per_epoch计算（需要从配置推断steps_per_epoch）
        # 这里假设已经在main.py中计算了steps_per_epoch
        max_epoch = getattr(cfg.SOLVER, 'MAX_EPOCH', 100)
        
        # 如果没有提供steps_per_epoch，尝试从配置文件获取或使用默认值
        if self.steps_per_epoch is not None:
            return max_epoch * self.steps_per_epoch
        else:
            # 使用合理的估算值
            default_total_iters = 10000
            print(f"Warning: Cannot auto-calculate total iterations, using default {default_total_iters}")
            return default_total_iters
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        if cfg.SOLVER.LR_POLICY.TYPE == 'Fix':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size = cfg.SOLVER.LR_POLICY.STEP_SIZE, 
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,  
                factor = cfg.SOLVER.LR_POLICY.PLATEAU_FACTOR, 
                patience = cfg.SOLVER.LR_POLICY.PLATEAU_PATIENCE
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Noam':
            self.scheduler = lr_scheduler.create(
                'Noam', 
                self.optimizer,
                model_size = cfg.SOLVER.LR_POLICY.MODEL_SIZE,
                factor = cfg.SOLVER.LR_POLICY.FACTOR,
                warmup = cfg.SOLVER.LR_POLICY.WARMUP,
                last_epoch = self.last_epoch
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'MultiStep':
            self.scheduler = lr_scheduler.create(
                'MultiStep', 
                self.optimizer,
                milestones = cfg.SOLVER.LR_POLICY.STEPS,
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Linear':
            # 自动计算总迭代数
            total_iters = self._calculate_total_iters()
            self.scheduler = lr_scheduler.create(
                'Linear',
                self.optimizer,
                total_iters = total_iters,
                start_factor = getattr(cfg.SOLVER.LR_POLICY, 'START_FACTOR', 1.0),
                end_factor = getattr(cfg.SOLVER.LR_POLICY, 'END_FACTOR', 0.0),
                last_epoch = self.last_epoch
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'CosineAnnealing':
            # 自动计算T_max（总迭代数）
            total_iters = self._calculate_total_iters()
            self.scheduler = lr_scheduler.create(
                'CosineAnnealing',
                self.optimizer,
                T_max = total_iters,
                eta_min = getattr(cfg.SOLVER.LR_POLICY, 'ETA_MIN', 0),
                last_epoch = self.last_epoch
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'CosineWarmRestarts':
            # 自动计算T_0（重启周期）
            total_iters = self._calculate_total_iters()
            # 如果配置中没有设置T_0，使用总迭代数的1/4作为默认值
            t_0 = getattr(cfg.SOLVER.LR_POLICY, 'T_0', total_iters // 4)
            self.scheduler = lr_scheduler.create(
                'CosineWarmRestarts',
                self.optimizer,
                T_0 = t_0,
                T_mult = getattr(cfg.SOLVER.LR_POLICY, 'T_MULT', 1),
                eta_min = getattr(cfg.SOLVER.LR_POLICY, 'ETA_MIN', 0),
                last_epoch = self.last_epoch
            )
        else:
            raise NotImplementedError

    def setup_optimizer(self, model):
        """
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR 
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            # 此处添加的"initial_lr"是为了NoamOpt恢复Epoch的lr
            params += [{"params": [value], "initial_lr": lr, "lr": lr, "weight_decay": weight_decay}]
        # """
        # 学习参数
        params = model.parameters()

        # 优化器设置
        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                momentum = cfg.SOLVER.SGD.MOMENTUM,
                nesterov = True
            )
        elif cfg.SOLVER.TYPE == 'ADAM':
            # 初始lr在scheduler为Noam时无效
            self.optimizer = torch.optim.Adam(
                params,
                lr = cfg.SOLVER.BASE_LR,
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(
                params, 
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RADAM':
            self.optimizer = RAdam(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        else:
            raise NotImplementedError

        # 学习率策略设置
        self._setup_scheduler()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, lrs_type, val=None):
        if self.scheduler is None:
            return

        if cfg.SOLVER.LR_POLICY.TYPE != 'Plateau':
            val = None

        if lrs_type == cfg.SOLVER.LR_POLICY.STEP_TYPE:
            self.scheduler.step(val)

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        return lr