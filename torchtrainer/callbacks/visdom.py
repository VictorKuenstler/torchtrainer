from visdom import Visdom
import numpy as np

from torchtrainer.callbacks.callbacks import Callback


class VisdomEpoch(Callback):
    def __init__(self, visdom_plotter, monitor='running_loss', on_iteration_every=100):
        """
        Plot your metrics on epoch end
        :param visdom_plotter: for example VisdomLinePlotter(env_name=f'Model {session_name}')
        :param monitor: the metric to plot (loss, running_loss)
        """
        super(VisdomEpoch, self).__init__()
        self.plotter = visdom_plotter
        self.monitor = monitor
        self.on_iteration_every = on_iteration_every
        self.iterations = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.plotter.plot(self.monitor, 'train', self.monitor, epoch, logs[self.monitor])
            self.plotter.plot(self.monitor, 'val', self.monitor, epoch, logs['val_' + self.monitor])

    def on_batch_end(self, iteration, logs=None):
        self.iterations += 1
        if self.iterations % self.on_iteration_every == 0:
            if logs is not None:
                self.plotter.plot(var_name=self.monitor + 'iteration', split_name='train',
                                  title_name=f'Iteration: {self.monitor}', x=self.iterations, y=logs[self.monitor],
                                  label='Iterations')


class VisdomIteration(Callback):
    def __init__(self, visdom_plotter, monitor='running_loss', on_iteration_every=100):
        """
        Plot your metrics on epoch end
        :param visdom_plotter: for example VisdomLinePlotter(env_name=f'Model {session_name}')
        :param monitor: the metric to plot (loss, running_loss)
        """
        super(VisdomIteration, self).__init__()
        self.plotter = visdom_plotter
        self.monitor = monitor
        self.on_iteration_every = on_iteration_every
        self.iterations = 0

    def on_iteration(self, iteration, logs=None):
        if logs is not None:
            self.plotter.plot(self.monitor, 'train', self.monitor, iteration, logs[self.monitor], label='iteration')
            self.plotter.plot(self.monitor, 'val', self.monitor, iteration, logs['val_' + self.monitor],
                              label='iteration')

    def on_batch_end(self, iteration, logs=None):
        self.iterations += 1
        if self.iterations % self.on_iteration_every == 0:
            if logs is not None:
                self.plotter.plot(var_name=self.monitor + 'iteration', split_name='train',
                                  title_name=f'Iteration: {self.monitor}', x=self.iterations, y=logs[self.monitor],
                                  label='Iterations')


class VisdomLinePlotter:
    def __init__(self, env_name='model'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, label='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=label,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')
