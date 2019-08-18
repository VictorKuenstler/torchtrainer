from tqdm import tqdm

from torchtrainer.callbacks.callbacks import Callback


def convert_logs(logs):
    log_metrics = {}
    for key, item in logs.items():
        if isinstance(item, int):
            item = '{:^10}'.format(item)
            key = '{:^10}'.format(key)
            log_metrics[key] = item
        else:
            item = '{:{}{}.{}}'.format(item, '^', 17, 4)
            key = '{:^17}'.format(key)
            log_metrics[key] = item
    return list(log_metrics.keys()), list(log_metrics.values())


class ProgressBar(Callback):
    """
    TQDM progress bar for training and validating
    """
    def __init__(self, log_every=100):
        """

        :param log_every: Number of iterations between writing log to console
        """
        self.training_bar = None
        self.validation_bar = None
        self.has_header = False
        self.has_val_header = False
        self.log_every = log_every
        self.train_logs = None
        super(ProgressBar, self).__init__()

    def __exit__(self, xtype, val, tb):
        if self.training_bar is not None:
            self.training_bar.close()
        if self.validation_bar is not None:
            self.validation_bar.close()

    def reset(self):
        self.has_val_header = False
        self.has_header = False

    def on_train_begin(self, logs=None):
        self.reset()
        self.train_logs = logs

    def on_epoch_begin(self, epoch, logs=None):
        self.training_bar = tqdm(total=self.train_logs['num_batches'], unit=' batch',
                                 desc=f'Training Epoch {epoch + 1}', position=0)

    def on_epoch_end(self, epoch, logs=None):
        logs['iteration'] = 00
        logs['epoch'] = epoch + 1
        keys, items = convert_logs(logs)

        if not self.has_header:
            tqdm.write('')
            tqdm.write(' '.join(keys))
            self.has_header = True

        tqdm.write(' '.join(items))

    def on_batch_begin(self, iteration, logs=None):
        self.training_bar.update(1)

    def on_batch_end(self, iteration, logs=None):
        if iteration % self.log_every == 0:
            logs['iteration'] = iteration + 1
            keys, items = convert_logs(logs)

            if not self.has_header:
                tqdm.write('')
                tqdm.write(' '.join(keys))
                self.has_header = True

            tqdm.write(' '.join(items))

    def on_validation_begin(self, logs=None):
        self.validation_bar = tqdm(total=self.train_logs['val_num_batches'], unit=' batch', desc='Validation',
                                   leave=True, position=1)

    def on_validation_epoch_begin(self, logs=None):
        self.validation_bar.update(1)

    def on_validation_end(self, logs=None):
        keys, items = convert_logs(logs)

        if not self.has_val_header:
            tqdm.write('Validation results')
            tqdm.write(' '.join(keys))
            self.has_val_header = True
        tqdm.write(' '.join(items))
        self.reset()
