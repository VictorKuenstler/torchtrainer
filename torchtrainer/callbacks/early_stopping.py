from torchtrainer.callbacks.callbacks import Callback


class EarlyStoppingEpoch(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.best_loss = float('inf')
        self.stopped_epoch = 0
        self.wait = 0
        super(EarlyStoppingEpoch, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)

        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait > self.patience:
                    self.stopped_epoch = epoch + 1
                    self.trainer.stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print(f'EarlyStopping terminated Training at Epoch {self.stopped_epoch}')


class EarlyStoppingIteration(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to stop the training after iteration_every
        :param monitor: 'loss', 'val_loss', 'running_loss', 'running_val_loss', every metric
        :param min_delta:
        :param patience: integer
        """
        self.iteration = 0
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.best_loss = float('inf')
        self.stopped_iteration = 0
        self.wait = 0
        super(EarlyStoppingIteration, self).__init__()

    def on_iteration(self, iteration, logs=None):
        current_loss = logs.get(self.monitor)
        self.iteration += 1

        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait > self.patience:
                    self.stopped_iteration = self.iteration + 1
                    self.trainer.stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_iteration > 0:
            print(f'EarlyStopping terminated Training at Epoch {self.stopped_iteration}')