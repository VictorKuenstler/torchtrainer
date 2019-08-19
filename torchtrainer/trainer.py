from copy import deepcopy

import torch.nn as nn
import torch

from torchtrainer.average_meter import AverageMeter
from torchtrainer.callbacks.callback_container import CallbackContainer
from torchtrainer.callbacks.metric_callback import MetricCallback
from torchtrainer.metrics.metric_container import MetricContainer
from torchtrainer.utils import check_loss, check_optimizer, check_loader


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
        self._val_metrics = []

        self._loss = None
        self._optimizer = None
        self._train_loader = None
        self._val_loader = None
        self._validate_every = None

        self._iterations = 0

        self.stop_training = False

        self.transform_fn = None
        self._batch_size = None

    def prepare(self, optimizer, loss, train_loader, val_loader, transform_fn=None, callbacks=[], metrics=[],
                validate_every=None):
        """
        Prepare the trainer
        :param optimizer: Optimization algorithm of your choice
        :param loss: Loss funciton of your choice
        :param train_loader: L
        :param val_loader:
        :param transform_fn: Function to transform batch into (*inputs, y_true) where inputs is tuple of inputs passed
        or (input_tensor, y_true)
        into your forward pass
        :param callbacks: List of callbacks
        :param metrics: List of metrics to log
        :param validate_every: Whether and when you want validate after x iteration (number of batches processed)
        :return:
        """
        self._set_optimzer(optimizer)
        self._set_loss(loss)

        self._set_train_loader(train_loader)
        self._set_validation(val_loader, validate_every)

        self._callbacks = callbacks
        self._metrics = metrics
        self._val_metrics = deepcopy(metrics)

        self._set_transform_fn(transform_fn)

    def train_loop(self, batch):
        """
        Implement your own train loop
        :param batch: batch returned by your DataLoader
        :return: y_pred: the predicted values, y_true: the true values, loss: the loss
        """
        self._optimizer.zero_grad()
        inputs, y_true = self.transform_fn(batch)

        y_pred = None
        if isinstance(inputs, torch.Tensor):
            y_pred = self.model(inputs)
        else:
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

        y_pred = None
        if isinstance(inputs, torch.Tensor):
            y_pred = self.model(inputs)
        else:
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
        self._batch_size = batch_size
        self.model.train()

        metrics = MetricContainer(self._metrics)

        container = CallbackContainer(self._callbacks)
        container.add(MetricCallback(metrics))
        container.set_trainer(self)

        container.on_train_begin({
            'batch_size': self._batch_size,
            'num_batches': len(self._train_loader),
            'val_num_batches': len(self._val_loader),
        })

        # running loss
        losses = AverageMeter('loss')
        final_result = {}

        for epoch in range(epochs):
            epoch_logs = {}
            container.on_epoch_begin(epoch, epoch_logs)

            for batch_idx, batch in enumerate(self._train_loader):
                batch_logs = {}
                container.on_batch_begin(self._iterations, batch_logs)

                # =================
                y_pred, y_true, loss = self.train_loop(batch)
                # =================

                losses.update(loss.item(), self._batch_size)

                batch_logs['loss'] = loss.item()
                batch_logs['running_loss'] = losses.avg
                batch_logs['iteration'] = self._iterations
                batch_logs.update(metrics(y_pred, y_true))

                if self._iteration_end_val():
                    val_logs = self.val()
                    batch_logs.update(val_logs)
                    container.on_iteration(self._iterations, batch_logs)
                    if self.stop_training:
                        break

                container.on_batch_end(self._iterations, batch_logs)
                epoch_logs.update(batch_logs)
            epoch_logs.update(self._epoch_end())
            final_result.update(epoch_logs)
            container.on_epoch_end(epoch, epoch_logs)
            losses.reset()
            if self.stop_training:
                break
        container.on_train_end(final_result)
        return final_result

    def val(self):
        """
        Call to start validation
        :return:
        """
        self.model.eval()
        check_loader(self._val_loader)
        metrics = MetricContainer(self._val_metrics)
        metrics.restart()

        container = CallbackContainer(self._callbacks)

        losses = AverageMeter('loss')
        validation_logs = {}

        container.on_validation_begin()
        for batch_idx, batch in enumerate(self._val_loader):
            container.on_validation_epoch_begin()
            batch_logs = {}
            y_pred, y_true, loss = self.val_loop(batch)

            losses.update(loss.item(), self._batch_size)
            batch_logs['loss'] = loss.item()
            batch_logs['running_loss'] = losses.avg
            batch_logs.update(metrics(y_pred, y_true))

            validation_logs.update(batch_logs)

        out_val_logs = {}
        for key, item in validation_logs.items():
            out_val_logs['val_' + key] = item

        container.on_validation_end(out_val_logs)

        # get back to train mode
        self.model.train()
        return out_val_logs

    def _set_loss(self, loss):
        self._loss = loss

    def _set_optimzer(self, optimizer):
        self._optimizer = optimizer

    def _set_train_loader(self, train_loader):
        self._train_loader = train_loader

    def _set_validation(self, val_loader, validate_every=None):
        self._val_loader = val_loader
        self._validate_every = validate_every

    def _set_transform_fn(self, fn):
        self.transform_fn = fn

    def _reset(self):
        """
        Resets the trainer for further training
        :return:
        """
        self._iterations = 0

    def _check(self):
        """
        Checks inputs to be correct for training
        :return:
        """
        check_loss(self._loss)
        check_optimizer(self._optimizer)
        check_loader(self._train_loader)

    def _iteration_end_val(self):
        """
        Checks whether validate_every is reached
        :return: boolean
        """
        self._iterations += 1
        if self._validate_every is not None and self._iterations % self._validate_every == 0:
            return True

    def _epoch_end(self):
        """
        Called after epoch end reached
        :return:
        """
        if self._validate_every is None:
            return self.val()
        else:
            return {}
