import glob
import os
import shutil

import torch

from torchtrainer.callbacks.callbacks import Callback


class Checkpoint(Callback):
    """
    Checkpointing after each iteration. Additionally saves the current best model in seperate file
    """
    def __init__(self, directory, filename='snapshot', best_filename='best', monitor='val_loss'):
        """

        :param directory: directory to save checkpoints ins
        :param filename: prefix for snapshots
        :param best_filename: filename for best model
        :param monitor: value to compare models
        """
        self.directory = directory
        self.filename = filename
        self.monitor = monitor
        self.best_filename = best_filename + '.pt'
        self.best = float('inf')

        super(Checkpoint, self).__init__()

    def save(self, epoch, file, is_best=False):
        torch.save(self.trainer.model, file)

        if is_best:
            shutil.copyfile(file, file.replace(self.filename, self.best_filename))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        values = f'epoch_{epoch + 1}_'
        for key, item in logs.items():
            if isinstance(item, int):
                values += f'{key}_{item}_'
            else:
                values += f'{key}_{"{:.2f}".format(item)}_'

        snapshot_prefix = os.path.join(self.directory, self.filename + '_')
        snapshot_path = snapshot_prefix + values + '.pt'

        current = logs.get(self.monitor)

        if current is None:
            pass
        elif current < self.best:
            self.best = current
            for f in glob.glob(os.path.join(self.directory, self.best_filename) + '*'):
                if f != snapshot_path:
                    os.remove(f)
            self.save(epoch, snapshot_path, True)
        else:
            self.save(epoch, snapshot_path)

        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)


class CheckpointIteration(Callback):
    def __init__(self, directory, filename='snapshot', best_filename='best', monitor='val_loss'):
        """

        :param directory: directory to save checkpoints ins
        :param filename: prefix for snapshots
        :param best_filename: filename for best model
        :param monitor: value to compare models
        """
        self.directory = directory
        self.filename = filename
        self.best_filename = best_filename + '.pt'
        self.monitor = monitor
        self.best = float('inf')

        super(CheckpointIteration, self).__init__()

    def save(self, iteration, file, is_best=False):
        torch.save(self.trainer.model, file)

        if is_best:
            shutil.copyfile(file, file.replace(self.filename, self.best_filename))

    def on_iteration(self, iteration, logs=None):
        logs = logs or {}
        values = f'epoch_{iteration + 1}_'
        for key, item in logs.items():
            if isinstance(item, int):
                values += f'{key}_{item}_'
            else:
                values += f'{key}_{"{:.2f}".format(item)}_'

        snapshot_prefix = os.path.join(self.directory, self.filename + '_')
        snapshot_path = snapshot_prefix + values + '.pt'

        current = logs.get(self.monitor)

        if current is None:
            pass
        elif current < self.best:
            self.best = current
            for f in glob.glob(os.path.join(self.directory, self.best_filename) + '*'):
                if f != snapshot_path:
                    os.remove(f)
            self.save(iteration, snapshot_path, True)
        else:
            self.save(iteration, snapshot_path)

        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)
