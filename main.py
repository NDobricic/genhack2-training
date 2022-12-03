import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch import nn

# import tester
import trainer

from generator import Generator
from error import Errors

from utils import *

props = {
    'device': torch.device("cuda"),
    'data_length': 8032,
    'latent_dim': 6,
    'batch_size': 32,
    'snapshot_at': 100,
    'gen_lr': 0.001,
    'disc_lr': 0.0008,
    'noise_scale': 0.03,
    'loss_func': nn.BCELoss(),
    'disc_step': 1,
    'gen_step': 5,
    'num_epochs': 300,
    'snapshot_interval': 100
}


# torch.manual_seed(111)

def load_data(file):
    columns = ["dates", "s1", "s2", "s3", "s4", "s5", "s6"]
    df = pd.read_csv(file, usecols=columns)

    data = np.asarray(df.iloc[:, 1:].values.tolist())

    return data


def train(data):
    date_time_str = datetime.now().strftime("%Y-%m-%d %H;%M;%S")
    return trainer.train(data, props, f'saved/{date_time_str}')


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


def generate(train_data_length, file):
    latent_variable = torch.randn(train_data_length, props['latent_dim'])

    generator = Generator(props['latent_dim'])
    generator.load_state_dict(torch.load(file))
    generator.eval()

    generated_samples = generator(latent_variable)
    generated_samples = generated_samples.detach().cpu().tolist()
    return generated_samples


def run():
    data = load_data('df_train.csv')
    gen_path = train(data)

    data_length = props['data_length']
    real_data = data[:data_length, :]
    generated_data = generate(data_length, gen_path)

    plot_results(real_data, 'Real')
    plot_results(generated_data, 'Generated')


def compare(ntest, nstart, file):
    data = load_data('df_train.csv')

    real_data = data[nstart:ntest+nstart, :]
    generated_data = generate(ntest, file)

    plot_results(real_data, 'Real')
    plot_results(generated_data, file)

    error = Errors(np.transpose(real_data), np.transpose(generated_data))

    print('Calculating marginal error...')
    marginal_error = error.marginal()
    print(marginal_error)

    print('Calculating dependency error...')
    dependency_error = error.dependency()
    print(dependency_error)
