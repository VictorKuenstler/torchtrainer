import csv

from torchtrainer.callbacks.callbacks import Callback


class CSVLogger(Callback):
    """
    Log after every epoch
    """

    def __init__(self,
                 file):
        """
        Init
        :param file: path to log file
        """
        super(CSVLogger, self).__init__()
        self.file = file
        self.columns = None
        self.writer = None
        self.csv_file = None

    def on_train_begin(self, logs=None):
        self.csv_file = open(self.file, 'w')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs.copy() or {}
        logs['epoch'] = epoch
        if not self.writer:
            self.columns = sorted(logs.keys())

            self.writer = csv.DictWriter(self.csv_file, fieldnames=[k for k in self.columns if k])
            self.writer.writeheader()

        self.writer.writerow(logs)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class CSVLoggerIteration(CSVLogger):
    """
    Log every time validation is run if you set up validate_every on your trainer
    """

    def __init__(self,
                 file):
        """
        Init
        :param file: path to log file
        """
        super(CSVLoggerIteration, self).__init__(file)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_iteration(self, iteration, logs=None):
        logs = logs.copy() or {}
        logs['iteration'] = iteration
        if not self.writer:
            self.columns = sorted(logs.keys())

            self.writer = csv.DictWriter(self.csv_file, fieldnames=[k for k in self.columns if k])
            self.writer.writeheader()

        self.writer.writerow(logs)
        self.csv_file.flush()
