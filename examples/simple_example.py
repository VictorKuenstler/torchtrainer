from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from tests.fixtures import Net
from torchtrainer.callbacks.checkpoint import Checkpoint
from torchtrainer.callbacks.csv_logger import CSVLogger
from torchtrainer.callbacks.early_stopping import EarlyStoppingEpoch
from torchtrainer.callbacks.progressbar import ProgressBar
from torchtrainer.callbacks.reducelronplateau import ReduceLROnPlateauCallback
from torchtrainer.callbacks.visdom import VisdomLinePlotter, VisdomEpoch
from torchtrainer.metrics.binary_accuracy import BinaryAccuracy
from torchtrainer.trainer import TorchTrainer


def transform_fn(batch):
    inputs, y_true = batch
    return inputs, y_true.float()


metrics = [BinaryAccuracy()]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_loader = DataLoader(FakeData(size=100, image_size=(3, 32, 32), num_classes=2, transform=transform),
                         batch_size=4,
                         shuffle=True,
                         num_workers=1)
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
    CSVLogger('test.log'),
    Checkpoint('./model'),
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
