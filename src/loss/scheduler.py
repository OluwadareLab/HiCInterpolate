from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_scheduler):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.finished_warmup = True
            return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        if self.finished_warmup:
            if epoch is None:
                self.after_scheduler.step(self.last_epoch - self.warmup_steps)
            else:
                self.after_scheduler.step(epoch - self.warmup_steps)
        else:
            return super().step(epoch)


class ExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps=10, decay_rate=0.96, staircase=True, last_epoch=-1, verbose=False):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        super(ExponentialDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.staircase:
            exponent = self.last_epoch // self.decay_steps
        else:
            exponent = self.last_epoch / self.decay_steps

        factor = self.decay_rate ** exponent
        return [base_lr * factor for base_lr in self.base_lrs]
