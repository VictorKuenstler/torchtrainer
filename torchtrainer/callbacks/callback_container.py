from torchtrainer.utils import current_time


class CallbackContainer:
    """
    Container holding all the callbacks
    """
    def __init__(self, callbacks=None, trainer=None):
        callbacks = callbacks or []

        self.callbacks = [callback for callback in callbacks]
        self.trainer = trainer

    def add(self, callback):
        self.callbacks.append(callback)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)
