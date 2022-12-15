import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime

import torch
from torch import nn

from error import anderson_darling, kendall
from gan import GAN

from utils import *
from vae import VAE

props = {
    'device': torch.device("cuda"),
    'data_start': 7000,
    'data_length': 2180,
    'latent_dim': 6,
    'batch_size': 16,
    'snapshot_at': 100,
    'gen_lr': 0.001,
    'disc_lr': 0.0008,
    'noise_scale': -0.01,
    'loss_func': nn.BCELoss(),
    'disc_step': 1,
    'gen_step': 1,
    'num_epochs': 1000,
    'snapshot_interval': 100
}


# torch.manual_seed(111)

def load_data(file):
    columns = ["s1", "s2", "s3", "s4", "s5", "s6"]
    df = pd.read_csv(file, usecols=columns)
    df = (df - df.mean()) / df.std()

    data = np.asarray(df.values.tolist())

    print(data)

    return data


def plot_results(plot_data, title):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(6):
        subplot = figure.add_subplot(2, 3, i + 1)
        subplot.set_ylim([0, 150])
        subplot.set_title(f'Station {i + 1}')
        plot_hist(subplot, [x[i] for x in plot_data])
    figure.tight_layout()

    plt.show()


def plot_results2(plot_data, title):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(6):
        subplot = figure.add_subplot(2, 3, i + 1)
        subplot.set_ylim([0, 300])
        subplot.set_title(f'Station {i + 1}')
        sb.histplot([x[i] for x in plot_data], kde=True)

    figure.tight_layout()

    plt.show()


def plot_results3(real, fake, title):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(6):
        subplot = figure.add_subplot(2, 3, i + 1)
        # subplot.set_ylim([0, 1])
        subplot.set_title(f'Station {i + 1}')
        sb.histplot([x[i] for x in real], kde=True, stat="density")
        sb.histplot([x[i] for x in fake], kde=True, stat="density")

    figure.tight_layout()

    plt.show()


def run():
    data = load_data('df_train.csv')
    date_time_str = datetime.now().strftime("%Y-%m-%d %H;%M;%S")
    gan = GAN(props, 6, f'saved\\{date_time_str}')
    gan.train(data)

    compare(gan, props['data_length'], props['data_start'])

    return gan


def compare(gan, num_samples, offset):
    data = load_data('df_train.csv')

    real_data = data[offset:num_samples + offset, :]
    generated_data = gan.generate(num_samples)

    print('Plotting results...')
    # plot_results2(real_data, 'Real')
    # plot_results2(generated_data, file)
    plot_results3(real_data, generated_data, str(gan))
    print(real_data)
    print(generated_data)

    print('Calculating marginal error...')
    marginal_error = anderson_darling(real_data, generated_data)
    print(f'Marginal error is {marginal_error}')

    print('Calculating dependency error...')
    dependency_error = kendall(real_data, generated_data)
    print(dependency_error)
