from torch import nn
from torch.optim.sgd import SGD

from tests.integration.utils import check_file_exists, remove_file, get_num_lines,\
    create_test_directory, delete_folder, get_num_files_in_directory
from torchtrainer.callbacks.checkpoint import Checkpoint, CheckpointIteration
from torchtrainer.callbacks.csv_logger import CSVLogger, CSVLoggerIteration
from torchtrainer.callbacks.progressbar import ProgressBar
from torchtrainer.metrics.binary_accuracy import BinaryAccuracy
from torchtrainer.callbacks.early_stopping import EarlyStoppingEpoch, EarlyStoppingIteration
from torchtrainer.trainer import TorchTrainer


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


def test_early_stopping_epoch(simple_neural_net):
    trainer = TorchTrainer(simple_neural_net)

    patience = 5

    early_stopping = EarlyStoppingEpoch('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 2):
        early_stopping.on_epoch_end(i, {'loss': 1})

    assert trainer.stop_training

    trainer = TorchTrainer(simple_neural_net)

    early_stopping = EarlyStoppingEpoch('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 1):
        early_stopping.on_epoch_end(i, {'loss': i})

    assert not trainer.stop_training


def test_early_stopping_iteration(simple_neural_net):
    trainer = TorchTrainer(simple_neural_net)

    patience = 5

    early_stopping = EarlyStoppingIteration('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 2):
        early_stopping.on_iteration(i, {'loss': 1})

    assert trainer.stop_training

    trainer = TorchTrainer(simple_neural_net)

    early_stopping = EarlyStoppingIteration('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 1):
        early_stopping.on_iteration(i, {'loss': i})

    assert not trainer.stop_training


def test_checkpointing(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    directory = './test_checkpointing'

    create_test_directory(directory)

    callbacks = [Checkpoint(directory)]

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    validate_every=1,
                    callbacks=callbacks)

    epochs = 4
    batch_size = 4
    trainer.train(epochs, batch_size)

    assert get_num_files_in_directory(directory) == 2

    delete_folder(directory)


def test_checkpointing_iteration(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    directory = './test_checkpointing'

    create_test_directory(directory)

    callbacks = [CheckpointIteration(directory)]

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    validate_every=1,
                    callbacks=callbacks)

    epochs = 4
    batch_size = 4
    trainer.train(epochs, batch_size)

    assert get_num_files_in_directory(directory) == 2

    delete_folder(directory)


def test_progressbar(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    callbacks = [ProgressBar(log_every=1)]

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    validate_every=1,
                    callbacks=callbacks)

    epochs = 4
    batch_size = 4
    trainer.train(epochs, batch_size)
