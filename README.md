# torchtrainer


PyTorch model training made simpler without loosing control. Focus on optimizing your model! Concepts are heavily inspired by the awesome project [torchsample](https://github.com/ncullen93/torchsample) and [Keras](https://github.com/keras-team/keras).
Further, besides applying Epoch Callbacks it also allows to call Callbacks every time after a specific number of batches passed (iterations) for long epoch durations.

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
from torchtrainer import TorchTrainer
from torchtrainer.callbacks import VisdomLinePlotter, ProgressBar, VisdomEpoch, Checkpoint, CSVLogger, \
    EarlyStoppingEpoch, ReduceLROnPlateauCallback
from torchtrainer.metrics import BinaryAccuracy


metrics = [BinaryAccuracy()]

train_loader = ...
val_loader = ...

model = ...
loss = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Setup Visdom Environment for your modl
plotter = VisdomLinePlotter(env_name=f'Model {11}')


# Setup the callbacks of your choice

callbacks = [
    ProgressBar(log_every=10),
    VisdomEpoch(plotter, on_iteration_every=10),
    VisdomEpoch(plotter, on_iteration_every=10, monitor='binary_acc'),
    CSVLogger('test.csv'),
    Checkpoint('./model'),
    EarlyStoppingEpoch(min_delta=0.1, monitor='val_running_loss', patience=10),
    ReduceLROnPlateauCallback(factor=0.1, threshold=0.1, patience=2, verbose=True)
]

trainer = TorchTrainer(model)

# function to transform batch into inputs to your model and y_true values
# if your model accepts multiple inputs, just put all inputs into a tuple (input1, input2), y_true
def transform_fn(batch):
    inputs, y_true = batch
    return inputs, y_true.float()

# prepare your trainer for training
trainer.prepare(optimizer,
                loss,
                train_loader,
                val_loader,
                transform_fn=transform_fn,
                callbacks=callbacks,
                metrics=metrics)

# train your model
result = trainer.train(epochs=10, batch_size=10)

``` 


### Callbacks

#### Logger

* `CSVLogger`
* `CSVLoggerIteration`
* `ProgressBar`

#### Visualization and Logging

* `VisdomEpoch`

#### Optimizers

* `ReduceLROnPlateauCallback`
* `StepLRCallback`

#### Regularization

* `EarlyStoppingEpoch`
* `EarlyStoppingIteration`

#### Checkpointing

* `Checkpoint`
* `CheckpointIteration`

### Metrics

Currently only `BinaryAccuracy` is implemented. To implement other Metrics use the abstract base metric class `torchtrainer.metrics.metric.Metric`. 

## TODO

- more tests
- metrics


## 

