from torch import nn
from torch.optim import SGD

from tests.fixtures import fake_loader, simple_neural_net
from torchtrainer.callbacks.progressbar import ProgressBar
from torchtrainer.modules.trainer import TorchTrainer


def transform_fn(batch):
    inputs, y_true = batch
    return inputs, y_true.float()

train_loader = fake_loader()
val_loader = fake_loader()

loss = nn.BCELoss()
optimizer = SGD(simple_neural_net.parameters(), lr=0.001, momentum=0.9)

callbacks = [ProgressBar(log_every=1)]

trainer = TorchTrainer(simple_neural_net())
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
