from math import log
import numpy as np


def anderson_darling(real_samples, generated_samples):
    real = np.transpose(real_samples)
    fake = np.sort(np.transpose(generated_samples))

    ntest = len(real[0])
    num_stations = len(real)

    u = np.zeros((num_stations, ntest))
    for s in range(num_stations):
        for i in range(ntest):
            u[s, i] = (sum([1 if real[s, j] <= fake[s, i] else 0 for j in range(ntest)]) + 1) / (ntest + 2)

    w = np.zeros(num_stations)
    for s in range(num_stations):
        w[s] = -ntest - sum(
            [(2 * (i + 1) - 1) * (log(u[s, i]) + log(1 - u[s, ntest - i - 1])) for i in range(ntest)]) / ntest

    return sum(w) / num_stations

def kendall(real_samples, generated_samples):
    real = np.transpose(real_samples)
    fake = np.transpose(generated_samples)

    ntest = len(real[0])
    num_stations = len(real)

    r = [sum([1 if all(
        [real[s, j] < real[s, i] for s in range(num_stations)]
    ) else 0 for j in range(ntest) if i != j]) / (ntest - 1) for i in range(ntest)]

    rt = [sum([1 if all(
        [fake[s, j] < fake[s, i] for s in range(num_stations)]
    ) else 0 for j in range(ntest) if i != j]) / (ntest - 1) for i in range(ntest)]

    r = np.sort(r)
    rt = np.sort(rt)

    return sum([abs(r[i] - rt[i]) for i in range(ntest)]) / ntest
