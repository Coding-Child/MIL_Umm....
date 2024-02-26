import torch
from torch.optim.lr_scheduler import _LRScheduler

class NoamLR(_LRScheduler):
    """The LR scheduler proposed by Noam

    This is simplified to use only torch components, removing external dependencies.

    Ref:
        "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 320,
        warmup_steps: int = 25000,
        last_epoch: int = -1,
    ):
        self.model_size = d_model
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate using Noam scheme."""
        step_num = self.last_epoch + 1
        return [
            lr * self.model_size**(-0.5) * 
            min(step_num**(-0.5), step_num * self.warmup_steps**(-1.5))
            for lr in self.base_lrs
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(model_size={self.model_size}, warmup_steps={self.warmup_steps})"