import torch
from torch import nn
import math
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import json
from json import JSONEncoder

from generator import Generator
from discriminator import Discriminator


def train(data, props, folder):
    generator = Generator(props['latent_dim']).to(device=props['device'])
    discriminator = Discriminator().to(device=props['device'])

    data_length = props['data_length'] - props['data_length'] % props['batch_size']
    data = data[:data_length, :]

    # prepare training set
    train_data = torch.zeros(data_length)

    train_data[:] = torch.Tensor(data[:])

    train_labels = torch.zeros(data_length)

    train_set = [(train_data[i], train_labels) for i in range(data_length)]

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=props['batch_size'], shuffle=True,
    )

    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=props['disc_lr'], betas=(0.5, 0.999))

    optimizer_generator = torch.optim.Adam(
        generator.parameters(), lr=props['gen_lr'], betas=(0.5, 0.999))

    start_time = time.time()

    for epoch in range(props['num_epochs']):

        for n, (real_samples, _) in enumerate(train_loader):

            real_samples = real_samples.to(device=props['device'])
            real_samples_labels = torch.ones(props['batch_size']).to(device=props['device'])

            # Data for training the discriminator

            if n % props['disc_step'] == 0:
                latent_space_samples = torch.randn(
                    (props['batch_size'], props['latent_dim'])
                ).to(device=props['device'])

                generated_samples = generator(latent_space_samples)

                generated_samples_labels = torch.zeros(props['batch_size']).to(device=props['device'])

                # real_samples_labels, generated_samples_labels = generated_samples_labels, real_samples_labels
                all_samples = torch.cat((real_samples, generated_samples))

                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = props['loss_func'](
                    output_discriminator, all_samples_labels
                )

                loss_discriminator.backward()
                optimizer_discriminator.step()

            if n % props['gen_step'] == 0:
                # Data for training the generator

                latent_space_samples = torch.randn(
                    (props['batch_size'], props['latent_dim'])
                ).to(device=props['device'])

                # Training the generator

                generator.zero_grad()

                generated_samples = generator(latent_space_samples)
                sample_noise = torch.rand(props['batch_size']).to(device=props['device'])

                output_discriminator_generated = discriminator(
                    generated_samples + sample_noise * props['noise_scale']
                ).to(device=props['device'])

                loss_generator = props['loss_func'](
                    output_discriminator_generated, real_samples_labels
                )

                loss_generator.backward()

                optimizer_generator.step()

            # Show loss

            if epoch % 10 == 0 and n == data_length // props['batch_size'] - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")

                print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    os.makedirs(folder)
    torch.save(generator.state_dict(), os.path.join(folder, 'gen'))
    torch.save(discriminator.state_dict(), os.path.join(folder, 'disc'))
    with open(os.path.join(folder, 'props.json'), 'w') as file:
        json.dump(props, file, cls=MyEncoder)

    print(f'Time taken: {time.time() - start_time}')


class MyEncoder(JSONEncoder):
    def default(self, o):
        # if type(o) == torch.device or type(o).__base__ == torch.nn.modules.loss._WeightedLoss:
        #    return str(o)
        return str(o)