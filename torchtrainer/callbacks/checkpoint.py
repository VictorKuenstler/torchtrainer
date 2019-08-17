import glob
import os
import shutil

import torch

from torchtrainer.callbacks.callbacks import Callback


class Checkpoint(Callback):
    def __init__(self, directory, filename='snapshot', monitor='val_loss', best_only=False):
        self.directory = directory
        self.filename = filename
        self.monitor = monitor
        self.best_only = best_only
        self.best = float('inf')

        super(Checkpoint, self).__init__()

    def save(self, epoch, file, is_best=False):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.trainer.model.state_dict(),
            'optimizer': self.trainer._optimizer.state_dict()
        }, file)

        if is_best:
            shutil.copyfile(file, 'model_best.pt')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        values = ''
        for key, item in logs.items():
            values += f'{key}_{item}_'

        snapshot_prefix = os.path.join(self.directory, self.filename)
        snapshot_path = snapshot_prefix + values + '.pt'
        if self.best_only:
            current = logs.get(self.monitor)
            if current is None:
                pass
            else:
                if current < self.best:
                    self.best = current
                    self.save(epoch, snapshot_path, True)
        else:
            self.save(epoch, snapshot_path)

        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)

