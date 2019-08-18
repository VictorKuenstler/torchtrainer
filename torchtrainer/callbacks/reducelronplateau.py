from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchtrainer.callbacks.callbacks import Callback


class ReduceLROnPlateauCallback(Callback):
    """
    ReduceLROnPlateu Callback
    """
    def __init__(self, on_iteration_every=None, monitor='running_loss', **kwargs):
        """

        :param on_iteration_every: Whether to call it after every X batches if None it is called after every epoch
        :param monitor: Monitor to compare on
        :param kwargs: Parameters passed to torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        self.scheduler = None
        super(ReduceLROnPlateauCallback, self).__init__()
        self.args = kwargs
        self.on_iteration_every = on_iteration_every
        self.monitor = monitor

    def on_train_begin(self, logs=None):
        self.scheduler = ReduceLROnPlateau(self.trainer._optimizer, **self.args)

    def on_epoch_end(self, epoch, logs=None):
        if self.on_iteration_every is None:
            current_monitor = logs.get(self.monitor)
            if current_monitor is not None:
                self.scheduler.step(current_monitor)

    def on_batch_end(self, iteration, logs=None):
        if self.on_iteration_every is not None and iteration % self.on_iteration_every == 0:
            current_monitor = logs.get(self.monitor)
            if current_monitor is not None:
                self.scheduler.step(current_monitor)
