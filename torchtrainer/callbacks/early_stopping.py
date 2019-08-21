from torchtrainer.callbacks.callbacks import Callback
from tqdm import tqdm


class EarlyStoppingEpoch(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to stop the training called after every epoch
        :param monitor: 'loss', 'val_loss', 'running_loss', 'running_val_loss', every metric you want
        :param min_delta:
        :param patience: integer
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.history = []

        self.best_loss = float('inf')
        self.stopped_epoch = 0
        self.wait = 0
        super(EarlyStoppingEpoch, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        self.history.append(current_loss)
        self.history = self.history[-self.patience:]

        if current_loss is None:
            pass
        else:
            if abs(current_loss - self.best_loss) > self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.trainer.stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            tqdm.write(
                f'EarlyStopping terminated Training at Epoch {self.stopped_epoch} as {self.monitor} ' +
                f'did not decrease by {self.min_delta} for over {self.patience} Epochs. ' +
                f'History: {", ".join(["{:.3f}".format(el) for el in self.history])}'
            )


class EarlyStoppingIteration(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to stop the training after iteration_every
        :param monitor: 'loss', 'val_loss', 'running_loss', 'running_val_loss', every metric you want
        :param min_delta:
        :param patience: integer
        """
        self.iteration = 0
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.history = []
        self.best_loss = float('inf')
        self.stopped_iteration = 0
        self.wait = 0
        super(EarlyStoppingIteration, self).__init__()

    def on_iteration(self, iteration, logs=None):
        current_loss = logs.get(self.monitor)
        self.iteration += 1
        self.history.append(current_loss)
        self.history = self.history[-self.patience:]

        if current_loss is None:
            pass
        else:
            if abs(current_loss - self.best_loss) > self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait > self.patience or self.best_loss not in self.history:
                    self.stopped_iteration = self.iteration + 1
                    self.trainer.stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_iteration > 0:
            tqdm.write(
                f'EarlyStopping terminated Training at Epoch {self.stopped_iteration} as {self.monitor}' +
                f'did not decrease by {self.min_delta} for over {self.patience} Epochs.' +
                f'History: {", ".join(["{:.3f}".format(el) for el in self.history])}'
            )
