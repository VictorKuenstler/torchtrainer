class Callback(object):
    """
    Abstract Base Callback Class
    """

    def __init__(self):
        self.params = None
        self.trainer = None
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, iteration, logs=None):
        pass

    def on_batch_end(self, iteration, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_iteration(self, iteration, logs=None):
        pass

    def on_validation_begin(self, logs=None):
        pass

    def on_validation_end(self, logs=None):
        pass

    def on_validation_epoch_begin(self, logs=None):
        pass
