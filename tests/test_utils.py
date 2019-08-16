from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData

from torchtrainer.modules.utils import _check_loader


def test_check_loader():
    dataset = FakeData()
    data_loader = DataLoader(dataset)

    assert _check_loader(data_loader) == True

    assert _check_loader('') == False
