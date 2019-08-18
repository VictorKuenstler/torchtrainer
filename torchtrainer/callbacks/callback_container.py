from torchtrainer.utils import current_time


class CallbackContainer:
    """
    Container to hold and call all callbacks
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

    def on_batch_begin(self, iteration, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(iteration, logs)

    def on_batch_end(self, iteration, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(iteration, logs)

    def on_validation_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_validation_end(logs)

    def on_validation_epoch_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_validation_epoch_begin(logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs['stop_time'] = current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_iteration(self, iteration, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_iteration(iteration, logs)
