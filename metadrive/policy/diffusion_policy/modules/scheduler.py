import math
import inspect

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        init_signature = inspect.signature(_LRScheduler.__init__)
        if "verbose" in init_signature.parameters:
            super(WarmupCosLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        else:
            super(WarmupCosLR, self).__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / max(self.warmup_epochs, 1)
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / max(self.warmup_epochs, 1)
        else:
            cosine_denominator = max(self.epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / cosine_denominator
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
