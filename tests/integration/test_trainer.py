from torch import nn
from torch.optim.sgd import SGD

from torchtrainer.metrics.binary_accuracy import BinaryAccuracy
from torchtrainer.modules.trainer import TorchTrainer


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
