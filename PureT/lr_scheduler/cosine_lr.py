import math
import torch.optim.lr_scheduler as lr_scheduler

class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    Anneals the learning rate using a cosine annealing schedule.
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_steps=0, warmup_init_lr=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.warmup_steps = max(0, int(warmup_steps))
        self.warmup_init_lr = warmup_init_lr
        # Effective cosine length after warmup (avoid zero division)
        self.T_max_eff = max(1, T_max - self.warmup_steps)
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize learning rates
        self.step(0)
    
    def _get_warmup_lr(self):
        # Linear warmup from warmup_init_lr to base_lr
        # Use (last_epoch + 1) so first step is already > warmup_init_lr
        progress = max(0, self.last_epoch + 1)
        denom = float(self.warmup_steps) if self.warmup_steps > 0 else 1.0
        factor = min(progress / denom, 1.0)
        return [self.warmup_init_lr + (base_lr - self.warmup_init_lr) * factor for base_lr in self.base_lrs]

    def _get_cosine_lr(self):
        # Cosine decay applied after warmup; use effective epoch offset
        t = max(0, self.last_epoch - self.warmup_steps)
        cosine = (1 + math.cos(math.pi * t / self.T_max_eff)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self._get_warmup_lr()
        return self._get_cosine_lr()
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts learning rate scheduler.
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, warmup_steps=0, warmup_init_lr=0.0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_steps = max(0, int(warmup_steps))
        self.warmup_init_lr = warmup_init_lr
        self.T_cur = last_epoch
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize learning rates
        self.step(0)
    
    def _get_warmup_lr(self):
        progress = max(0, self.last_epoch + 1)
        denom = float(self.warmup_steps) if self.warmup_steps > 0 else 1.0
        factor = min(progress / denom, 1.0)
        return [self.warmup_init_lr + (base_lr - self.warmup_init_lr) * factor for base_lr in self.base_lrs]

    def _get_cosine_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self._get_warmup_lr()
        return self._get_cosine_lr()
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.last_epoch < self.warmup_steps:
            # During warmup just update T_cur for completeness
            self.T_cur = 0
        else:
            # offset epoch by warmup
            t_epoch = self.last_epoch - self.warmup_steps
            if t_epoch >= self.T_i:
                if self.T_mult == 1:
                    self.T_cur = t_epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((t_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = t_epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0 if self.T_mult == 1 else self.T_i
                self.T_cur = t_epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
