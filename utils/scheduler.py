import math
from torch.optim.lr_scheduler import _LRScheduler


def update_lr_warm_up(nb_iter, optimizer, warm_up_iter, lr):
    """Legacy warmup function - kept for backward compatibility"""
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return current_lr


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        warmup_start_lr: Initial learning rate for warmup (default: 1e-6)
        min_lr: Minimum learning rate after decay (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 warmup_start_lr=1e-6, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                    for base_lr in self.base_lrs]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and linear decay.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        warmup_start_lr: Initial learning rate for warmup (default: 1e-6)
        min_lr: Minimum learning rate after decay (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 warmup_start_lr=1e-6, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return [self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                    for base_lr in self.base_lrs]


def get_scheduler(config, optimizer):
    """
    Get learning rate scheduler based on config.

    Args:
        config: Configuration object with scheduler settings
        optimizer: Optimizer to schedule

    Returns:
        Learning rate scheduler
    """
    lr_scheduler_type = config.get('lr_scheduler_type', 'multistep')

    if lr_scheduler_type == 'warmup':
        # Warmup with cosine decay
        warmup_steps = config.get('warmup_steps', 1000)
        warmup_start_lr = config.get('warmup_start_lr', 1e-6)
        min_lr = config.get('min_lr', 1e-6)
        total_steps = config.max_steps

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            warmup_start_lr=warmup_start_lr,
            min_lr=min_lr
        )
        return scheduler, 'step'  # Return 'step' to indicate per-step scheduling

    elif lr_scheduler_type == 'warmup_linear':
        # Warmup with linear decay
        warmup_steps = config.get('warmup_steps', 1000)
        warmup_start_lr = config.get('warmup_start_lr', 1e-6)
        min_lr = config.get('min_lr', 1e-6)
        total_steps = config.max_steps

        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            warmup_start_lr=warmup_start_lr,
            min_lr=min_lr
        )
        return scheduler, 'step'

    elif lr_scheduler_type == 'multistep':
        # Original multistep scheduler
        milestones_ratios = config.get('milestones_ratios', [0.5, 0.7, 0.85])
        milestones = [int(config.max_steps * ratio) for ratio in milestones_ratios]
        gamma = config.get('gamma', 0.1)

        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return scheduler, 'step'

    else:
        raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")
