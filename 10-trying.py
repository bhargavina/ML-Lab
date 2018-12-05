import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

def local_regression(x0, X, Y, tau):
        x0 = np.r_[1, x0]
        X = np.c_[np.ones(len(X)), X]
        xw = X.T * np.exp(np.sum((X - x0) ** 2, axis = 1) / (-2 * tau ** 2))
        beta = np.linalg.pinv(xw @ X) @ xw @ Y
        return x0 @ beta

X = np.linspace(-3, 3, num = 1000)
Y = np.log(np.abs(X ** 2 - 1) + 0.5)
X += np.random.normal(scale = 0.1, size = 1000)
domain = np.linspace(-3, 3, num = 300)

def plot_lwr(tau):
        predictions = [local_regression(x0, X, Y, tau) for x0 in domain]
        plot = figure(plot_width = 400, plot_height = 400)
        plot.title.text = 'tau: %g' % tau
        plot.scatter(X, Y, alpha = 0.3)
        plot.line(domain, predictions, line_width = 2, color = 'red')
        return plot

show(gridplot([[plot_lwr(10), plot_lwr(1)], [plot_lwr(0.1), plot_lwr(0.01)]]))
        
