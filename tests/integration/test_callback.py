from torch import nn
from torch.optim.sgd import SGD

from tests.integration.utils import check_file_exists, remove_file, get_num_lines
from torchtrainer.callbacks.csv_logger import CSVLogger, CSVLoggerIteration
from torchtrainer.metrics.binary_accuracy import BinaryAccuracy
from torchtrainer.modules.trainer import TorchTrainer


def transform_fn(batch):
    inputs, y_true = batch
    return inputs, y_true.float()


def test_csv_logger(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    metrics = [BinaryAccuracy()]

    file = './test_log.csv'
    callbacks = [CSVLogger(file)]

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    metrics=metrics,
                    callbacks=callbacks,
                    validate_every=1)

    epochs = 1
    trainer.train(epochs, 4)

    assert check_file_exists(file)

    assert get_num_lines(file) == epochs + 1

    remove_file(file)


def test_csv_logger_iteration(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    metrics = [BinaryAccuracy()]

    file = './test_log.csv'
    callbacks = [CSVLoggerIteration(file)]

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    metrics=metrics,
                    callbacks=callbacks,
                    validate_every=1)

    epochs = 1
    batch_size = 4
    trainer.train(epochs, batch_size)

    assert check_file_exists(file)

    assert get_num_lines(file) == len(fake_loader) + 1

    remove_file(file)
