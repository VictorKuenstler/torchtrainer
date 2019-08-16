from tests.fixtures import Net
from torchtrainer.callbacks.early_stopping import EarlyStoppingEpoch, EarlyStoppingIteration
from torchtrainer.modules.trainer import TorchTrainer


def test_early_stopping_epoch():
    trainer = TorchTrainer(Net())

    patience = 5

    early_stopping = EarlyStoppingEpoch('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 2):
        early_stopping.on_epoch_end(i, {'loss': 1})

    assert trainer.stop_training == True

    trainer = TorchTrainer(Net())

    early_stopping = EarlyStoppingEpoch('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 1):
        early_stopping.on_epoch_end(i, {'loss': i})

    assert trainer.stop_training == False


def test_early_stopping_iteration():
    trainer = TorchTrainer(Net())

    patience = 5

    early_stopping = EarlyStoppingIteration('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 2):
        early_stopping.on_iteration(i, {'loss': 1})

    assert trainer.stop_training == True

    trainer = TorchTrainer(Net())

    early_stopping = EarlyStoppingIteration('loss', min_delta=0.1, patience=patience)
    early_stopping.set_trainer(trainer)

    for i in range(patience + 1):
        early_stopping.on_iteration(i, {'loss': i})

    assert trainer.stop_training == False
