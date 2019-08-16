class Metric:
    def __init__(self, name):
        self.name = name

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Metric need to be correctly implemented')

    def restart(self):
        raise NotImplementedError('Metric need to be correctly implemented')
