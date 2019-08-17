from visdom import Visdom
import numpy as np

from torchtrainer.callbacks.callbacks import Callback


class VisdomEpochCallback(Callback):
    def __init__(self, visdom_plotter, monitor='running_loss'):
        """
        Plot your metrics on epoch end
        :param visdom_plotter: for example VisdomLinePlotter(env_name=f'Model {session_name}')
        :param monitor: the metric to plot (loss, running_loss)
        """
        super(VisdomEpochCallback, self).__init__()
        self.plotter = visdom_plotter
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.plotter.plot('loss', 'train', 'Loss', epoch, logs[self.monitor])


class VisdomLinePlotter:
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
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

    def plot_config(self, config):
        inner_table = ''
        for key, value in config.items():
            inner_table += f'<tr><th>{key}</th><td>{value}</td></tr>'
        config_html = f'<table border="1">{inner_table}</table>'
        self.viz.text(config_html, env=self.env)
