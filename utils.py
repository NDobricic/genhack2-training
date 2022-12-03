import numpy as np
import matplotlib.pyplot as plt


def plot_hist(subplot, data, rng=5, delta=None):
    if delta is None:
        delta = 0.3 / (len(data) / 512)
    a = int(rng // delta + 1)
    bins = [(x - a / 2) * delta for x in range(a)]
    hist, bin_edges = np.histogram(data, bins=bins)

    subplot.hist(data, bins=bin_edges)
