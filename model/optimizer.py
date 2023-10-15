import numpy as np
import torch

class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, model, constants, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=constants.BETAS,
            eps=constants.EPS,
            weight_decay=constants.WEIGHT_DECAY,
        )
        self.n_warmup_steps = constants.WARM_UP_STEP
        self.anneal_steps = constants.ANNEAL_STEP
        self.anneal_rate = constants.ANNEAL_RATE
        self.current_step = current_step
        self.init_lr = np.power(constants.ENCODER_HIDDEN, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
