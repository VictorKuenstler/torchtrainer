import torch.nn as nn


class TorchTrainer:
    def __init__(self, model):
        if not isinstance(model, nn.Module):
            raise ValueError('model is not instance of torch.nn.Module')

        self.model = model

        self._callbacks = []

        self._loss = None

        self._optimizer = None

    def set_loss(self, loss):
        self._loss = loss

    def set_optimzer(self, optimizer, **kwargs):
        if 'parameters' in kwargs:
            parameters = kwargs['parameters']
        else:
            parameters = self.model.parameters()

        self._optimizer = optimizer(parameters, **kwargs)

    def compile(self, optimizer, loss, callbacks=None):
        self.set_optimzer(optimizer)
        self.set_loss(loss)

    def train_loop(self, batch):
        pass

    def train(self, epochs):
        for epoch in range(epochs):

            self.train_loop()
