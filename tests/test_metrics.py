import torch

from torchtrainer.metrics.binary_accuracy import BinaryAccuracy


def test_binary_acc():
    acc_metric = BinaryAccuracy()

    y_true = torch.squeeze(torch.Tensor([0, 1, 1, 0]), -1)
    y_pred = torch.Tensor([0.2, 0.8, 0.2, 0.1])

    acc = acc_metric(y_pred, y_true)
    assert acc == 75.0

    y_true = torch.squeeze(torch.Tensor([0, 1, 1, 0]), -1)
    y_pred = torch.Tensor([0.2, 0.8, 0.2, 0.8])

    acc = acc_metric(y_pred, y_true)

    assert acc == 62.5
