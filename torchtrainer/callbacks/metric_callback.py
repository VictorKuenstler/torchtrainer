from torchtrainer.callbacks.callbacks import Callback


class MetricCallback(Callback):
    def __init__(self, container):
        self.container = container

    def on_epoch_begin(self, epoch, logs):
        self.container.restart()
