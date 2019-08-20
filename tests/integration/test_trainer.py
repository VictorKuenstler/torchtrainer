from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from tests.fixtures import Net
from torchtrainer.callbacks import StepLREpochCallback
from torchtrainer.callbacks.early_stopping import EarlyStoppingEpoch
from torchtrainer.callbacks.progressbar import ProgressBar
from torchtrainer.callbacks.reducelronplateau import ReduceLROnPlateauCallback
from torchtrainer.callbacks.visdom import VisdomLinePlotter, VisdomEpoch, VisdomIteration
from torchtrainer.metrics.binary_accuracy import BinaryAccuracy
from torchtrainer.trainer import TorchTrainer


def transform_fn(batch):
    inputs, y_true = batch
    return inputs, y_true.float()


def test_trainer_train_without_plugins(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer, loss, train_loader, val_loader, transform_fn=transform_fn)
    trainer.train(1, 4)


def test_trainer_train_with_metric(fake_loader, simple_neural_net):
    train_loader = fake_loader
    val_loader = fake_loader

    metrics = [BinaryAccuracy()]

    loss = nn.BCELoss()
    optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

    trainer = TorchTrainer(simple_neural_net)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    metrics=metrics,
                    validate_every=1)
    final_result = trainer.train(1, 4)

    assert 'binary_acc' in final_result
    assert 'val_binary_acc' in final_result


def test_trainer_train_iteration(fake_loader, simple_neural_net):
    def transform_fn(batch):
        inputs, y_true = batch
        return inputs, y_true.float()

    metrics = [BinaryAccuracy()]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = DataLoader(FakeData(size=100, image_size=(3, 32, 32), num_classes=2, transform=transform),
                              batch_size=4,
                              shuffle=True,
                              num_workers=1)
    val_loader = DataLoader(FakeData(size=50, image_size=(3, 32, 32), num_classes=2, transform=transform), batch_size=4,
                            shuffle=True,
                            num_workers=1)

    model = Net()
    loss = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    plotter = VisdomLinePlotter(env_name=f'Model {11}')

    callbacks = [
        ProgressBar(log_every=10),
        VisdomIteration(plotter),
        ReduceLROnPlateauCallback(factor=0.1, threshold=0.1, patience=2, verbose=True)
    ]

    trainer = TorchTrainer(model)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    callbacks=callbacks,
                    metrics=metrics,
                    validate_every=25)

    epochs = 10
    batch_size = 10
    trainer.train(epochs, batch_size)
    VisdomIteration(plotter, on_iteration_every=10, monitor='binary_acc'),


def test_trainer_train_full(fake_loader, simple_neural_net):
    def transform_fn(batch):
        inputs, y_true = batch
        return inputs, y_true.float()

    metrics = [BinaryAccuracy()]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = DataLoader(FakeData(size=100, image_size=(3, 32, 32), num_classes=2, transform=transform),
                              batch_size=4,
                              shuffle=True,
                              num_workers=1)
    val_loader = DataLoader(FakeData(size=50, image_size=(3, 32, 32), num_classes=2, transform=transform), batch_size=4,
                            shuffle=True,
                            num_workers=1)

    model = Net()
    loss = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    plotter = VisdomLinePlotter(env_name=f'Model {11}')

    callbacks = [
        ProgressBar(log_every=10),
        VisdomEpoch(plotter, on_iteration_every=10),
        VisdomEpoch(plotter, on_iteration_every=10, monitor='binary_acc'),
        EarlyStoppingEpoch(min_delta=0.1, monitor='val_running_loss', patience=10),

        ReduceLROnPlateauCallback(factor=0.1, threshold=0.1, patience=2, verbose=True)
    ]

    trainer = TorchTrainer(model)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    callbacks=callbacks,
                    metrics=metrics)

    epochs = 10
    batch_size = 10
    trainer.train(epochs, batch_size)


def test_trainer_train_steplr(fake_loader, simple_neural_net):
    def transform_fn(batch):
        inputs, y_true = batch
        return inputs, y_true.float()

    metrics = [BinaryAccuracy()]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = DataLoader(FakeData(size=100, image_size=(3, 32, 32), num_classes=2, transform=transform),
                              batch_size=4,
                              shuffle=True,
                              num_workers=1)
    val_loader = DataLoader(FakeData(size=50, image_size=(3, 32, 32), num_classes=2, transform=transform), batch_size=4,
                            shuffle=True,
                            num_workers=1)

    model = Net()
    loss = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    callbacks = [
        StepLREpochCallback()
    ]

    trainer = TorchTrainer(model)
    trainer.prepare(optimizer,
                    loss,
                    train_loader,
                    val_loader,
                    transform_fn=transform_fn,
                    callbacks=callbacks,
                    metrics=metrics)

    epochs = 10
    batch_size = 10
    trainer.train(epochs, batch_size)
