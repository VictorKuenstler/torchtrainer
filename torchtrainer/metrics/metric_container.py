class MetricContainer:
    def __init__(self, metrics):
        self.metrics = metrics

    def restart(self):
        for metric in self.metrics:
            metric.restart()

    def __call__(self, y_pred, y_true):
        logs = {}

        for metric in self.metrics:
            logs[metric.name] = metric(y_pred, y_true)

        return logs
