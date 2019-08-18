from torchtrainer.metrics.metric import Metric


class BinaryAccuracy(Metric):
    """
    Binary Running Accuracy Metric
    """
    def __init__(self):
        self.name = 'binary_acc'

        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = y_pred.round()
        self.correct_count += y_pred_round.eq(y_true).float().sum().item()
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy

    def restart(self):
        self.correct_count = 0
        self.total_count = 0
