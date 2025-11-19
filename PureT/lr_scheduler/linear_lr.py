import torch.optim.lr_scheduler as lr_scheduler

class LinearLR:
    """
    Linear learning rate scheduler.
    Linearly decreases the learning rate from start_factor to end_factor over total_iters.
    """
    def __init__(self, optimizer, total_iters, start_factor=1.0, end_factor=0.0, last_epoch=-1):
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.last_epoch = last_epoch
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize learning rates
        self.step(0)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        
        # Linear interpolation
        factor = self.start_factor + (self.end_factor - self.start_factor) * (self.last_epoch / self.total_iters)
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr