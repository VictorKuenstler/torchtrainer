import pytest
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData

from tests.fixtures import Net
from torchtrainer.modules.utils import check_loader, check_loss, check_optimizer


def test_check_loader():
    dataset = FakeData()
    data_loader = DataLoader(dataset)

    check_loader(data_loader)

    with pytest.raises(TypeError):
        check_loader(None)


def test_check_loss():
    loss = nn.BCELoss()

    check_loss(loss)

    with pytest.raises(TypeError):
        check_loader(None)


def test_check_optimizer():
    model = Net()
    optimizer = optim.Adam(model.parameters())

    check_optimizer(optimizer)

    with pytest.raises(TypeError):
        check_loader(None)
