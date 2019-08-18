from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from torchtrainer.callbacks.callbacks import Callback


class StepLREpochCallback(Callback):
    def __init__(self, step_size=30, gamma=0.1, on_iteration_every=None):
        """

        :param step_size: StepLR parameter
        :param gamma: StepLR parameter
        :param on_iteration_every: Whether to call it after every X batches if None it is called after every epoch
        """
        super(StepLREpochCallback, self).__init__()
        self.scheduler = None
        self.step_size = step_size
        self.gamma = gamma
        self.last_lr = -1
        self.on_iteration_every = on_iteration_every

    def on_train_begin(self, logs=None):
        self.scheduler = StepLR(self.trainer._optimizer, step_size=self.step_size, gamma=self.gamma)

    def on_epoch_end(self, epoch, logs=None):
        if self.on_iteration_every is None:
            self.scheduler.step()
            self.check_if_changed()

    def on_batch_end(self, iteration, logs=None):
        if self.on_iteration_every is not None and iteration % self.on_iteration_every == 0:
            self.scheduler.step()
            self.check_if_changed()

    def check_if_changed(self):
        current_lr = self.scheduler.get_lr()[0]

        if self.last_lr != -1:
            if self.last_lr != current_lr:
                self.last_lr = current_lr
                tqdm.write(f'StepLR changed Learning Rate to {self.last_lr}')
        else:
            self.last_lr = current_lr
