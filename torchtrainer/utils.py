import datetime
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader


def current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


def check_loader(loader):
    if not isinstance(loader, DataLoader):
        raise TypeError('loader needs to be instance of torch.utils.data.dataloader.DataLoader')


def check_optimizer(optimizer):
    if not isinstance(optimizer, Optimizer):
        raise TypeError('Loss needs to be instance of torch.optim.optimizer.Optimizer')


def check_loss(loss):
    if not isinstance(loss, _Loss):
        raise TypeError('Optimizer needs to be instance of torch.nn.modules.loss._Loss')
