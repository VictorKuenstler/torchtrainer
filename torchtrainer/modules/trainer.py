import torch.nn as nn

from torchtrainer.average_meter import AverageMeter
from torchtrainer.callbacks.callback_container import CallbackContainer
from torchtrainer.callbacks.metric_callback import MetricCallback
from torchtrainer.metrics.metric_container import MetricContainer
from torchtrainer.modules.utils import check_loss, check_optimizer, check_loader


class TorchTrainer:
    """
    Focus on optimizing your model and not on logging
    """

    def __init__(self, model):
        if not isinstance(model, nn.Module):
            raise ValueError('model is not instance of torch.nn.Module')

        self.model = model

        self._callbacks = []
        self._metrics = []

        self._loss = None
        self._optimizer = None
        self._train_loader = None
        self._val_loader = None
        self._validate_every = None

        self._iterations = 0

        self.stop_training = False

        self.transform_fn = None

    def set_loss(self, loss):
        self._loss = loss

    def set_optimzer(self, optimizer, **kwargs):
        if 'parameters' in kwargs:
            parameters = kwargs['parameters']
        else:
            parameters = self.model.parameters()

        self._optimizer = optimizer(parameters, **kwargs)

    def set_train_loader(self, train_loader):
        self._train_loader = train_loader

    def set_validation(self, val_loader, validate_every=None):
        self._val_loader = val_loader
        self._validate_every = validate_every

    def compile(self, optimizer, loss, train_loader, val_loader, callbacks=None, metrics=None, validate_every=None):
        self.set_optimzer(optimizer)
        self.set_loss(loss)

        self.set_train_loader(train_loader)
        self.set_validation(val_loader, validate_every)

        self._callbacks = callbacks
        self._metrics = metrics

    def train_loop(self, batch):
        """
        Implement your own train loop
        :param batch: batch returned by your DataLoader
        :return: y_pred: the predicted values, y_true: the true values, loss: the loss
        """
        self._optimizer.zero_grad()
        inputs, y_true = self.transform_fn(batch)
        y_pred = self.model(*inputs)

        loss = self._loss(y_pred, y_true)
        loss.backward()

        self._optimizer.step()

        return y_pred, y_true, loss

    def val_loop(self, batch):
        """
        Implement your own train loop
        :param batch: batch returned by your DataLoader
        :return: y_pred: the predicted values, y_true: the true values, loss: the loss
        """
        inputs, y_true = self.transform_fn(batch)
        y_pred = self.model(*inputs)

        loss = self._loss(y_pred, y_true)
        return y_pred, y_true, loss

    def train(self, epochs, batch_size):
        """
        Call to start your training
        :param epochs: number of epochs to train
        :param batch_size: your batch_size
        :return:
        """
        self._check()
        self._reset()

        self.model.train()

        metrics = MetricContainer(self._metrics)

        container = CallbackContainer(self._callbacks)
        container.add(MetricCallback(metrics))
        container.set_trainer(self)

        container.on_train_begin({
            'batch_size': batch_size,
            'num_batches': len(self.train_loader)
        })

        # running loss
        losses = AverageMeter('loss')

        for epoch in range(epochs):
            epoch_logs = {}
            container.on_epoch_begin(epoch, epoch_logs)

            for batch_idx, batch in enumerate(self.train_loader):
                batch_logs = {}
                container.on_epoch_begin(self._iterations, batch_logs)

                # =================
                y_pred, y_true, loss = self.train_loop(batch)
                # =================

                losses.update(loss.item(), batch_size)

                batch_logs['loss'] = loss.item()
                batch_logs['running_loss'] = losses.avg
                batch_logs.update(metrics(y_pred, y_true))

                container.on_batch_end(self._iterations, batch_logs)

                if self._iteration_end_val():
                    self.val()
                    container.on_iteration(self._iterations, batch_logs)
                    if self.stop_training:
                        break

                epoch_logs.update(batch_logs)

            self._epoch_end()
            container.on_epoch_end(epoch, epoch_logs)
            losses.reset()

            if self.stop_training:
                break

    def val(self):
        """
        Call to start validation
        :return:
        """
        self.model.eval()
        check_loader(self._val_loader)
        metrics = MetricContainer(self._metrics)

        for batch_idx, batch in enumerate(self._val_loader):
            y_pred, y_true, loss = self.val_loop(batch)
            metrics(y_pred, y_true)

    def _reset(self):
        self._iterations = 0

    def _check(self):
        check_loss(self._loss)
        check_optimizer(self._optimizer)
        check_loader(self._train_loader)

    def _iteration_end_val(self):
        self._iterations += 1
        if self._iterations % self._validate_every == 0:
            return True

    def _epoch_end(self):
        if self._validate_every is None:
            self.val()
