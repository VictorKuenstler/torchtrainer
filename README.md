# torchtrainer


PyTorch model training made simpler. Focus on optimizing your model! Concepts are heavily inspired by the awesome project [torchsample](https://github.com/ncullen93/torchsample) and [Keras](https://github.com/keras-team/keras). 

[![Build Status](https://travis-ci.com/VictorKuenstler/torchtrainer.svg?branch=master)](https://travis-ci.com/VictorKuenstler/torchtrainer)
[![codecov](https://codecov.io/gh/VictorKuenstler/torchtrainer/branch/master/graph/badge.svg)](https://codecov.io/gh/VictorKuenstler/torchtrainer)

## Features

* `Torchtrainer`
* Logging utilities
* Metrics
* Visdom Visualization
* Learning Rate Scheduler
* Checkpointing
* Flexible for muliple data inputs
* Setup validation after every ... batches

## Usage

### Installation

```bash
pip install torchtrainer
```


### Example

```python
from torch import nn
from torch.optim import SGD
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

train_loader = ...
val_loader = ...

model = ...
loss = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Setup Visdom Environment for your modl
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

# train your model
trainer.train(epochs=10, batch_size=10)
``` 

## TODO

- more tests
- metrics


## 

