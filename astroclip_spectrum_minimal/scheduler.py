import math

from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWithWarmupLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, T_warmup: int = 0, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            warmup_denom = max(self.T_warmup, 1)
            return [base_lr * self.last_epoch / warmup_denom for base_lr in self.base_lrs]
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]

        i = self.last_epoch - self.T_warmup
        n = max(self.T_max - self.T_warmup, 1)
        ratio = i / n
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return [self.eta_min + coeff * (base_lr - self.eta_min) for base_lr in self.base_lrs]
