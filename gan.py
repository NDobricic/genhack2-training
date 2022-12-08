from torch import nn
import torch
import numpy as np

import time
import os
import json
from json import JSONEncoder

from generator import Generator
from discriminator import Discriminator


class GAN:
    def __init__(self, props, num_stations, save_folder):
        self.folder = save_folder
        self.props = props
        self.num_stations = num_stations
        self.generator = Generator(props['latent_dim'], num_stations).to(device=props['device'])
        self.discriminator = Discriminator(num_stations).to(device=props['device'])

    def __str__(self):
        return f'GAN at folder {self.folder}'

    def train(self, data):
        print(f'Training {self}')
        data_length = self.props['data_length'] - self.props['data_length'] % self.props['batch_size']
        data = data[:data_length, :]

        # prepare training set
        train_data = torch.zeros((data_length, self.num_stations))

        for i in range(self.num_stations):
            train_data[:, i] = torch.Tensor(data[:, i])

        train_labels = torch.zeros((data_length, self.num_stations))

        train_set = [(train_data[i, :], train_labels[i, :]) for i in range(data_length)]

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.props['batch_size'], shuffle=True,
        )

        optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.props['disc_lr'], betas=(0.5, 0.999))

        optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=self.props['gen_lr'], betas=(0.5, 0.999))

        start_time = time.time()

        for epoch in range(self.props['num_epochs']):

            for n, (real_samples, _) in enumerate(train_loader):

                real_samples = real_samples.to(device=self.props['device'])
                real_samples_labels = torch.ones((self.props['batch_size'], self.num_stations)).to(device=self.props['device'])

                # Data for training the discriminator

                if n % self.props['disc_step'] == 0:
                    latent_space_samples = torch.randn(
                        (self.props['batch_size'], self.props['latent_dim'])
                    ).to(device=self.props['device'])

                    generated_samples = self.generator(latent_space_samples)

                    generated_samples_labels = torch.zeros(
                        (self.props['batch_size'], self.num_stations)
                    ).to(device=self.props['device'])

                    # real_samples_labels, generated_samples_labels = generated_samples_labels, real_samples_labels
                    all_samples = torch.cat((real_samples, generated_samples))

                    all_samples_labels = torch.cat(
                        (real_samples_labels, generated_samples_labels)
                    )

                    self.discriminator.zero_grad()
                    output_discriminator = self.discriminator(all_samples)
                    loss_discriminator = self.props['loss_func'](
                        output_discriminator, all_samples_labels
                    )

                    loss_discriminator.backward()
                    optimizer_discriminator.step()

                if n % self.props['gen_step'] == 0:
                    # Data for training the generator

                    latent_space_samples = torch.randn(
                        (self.props['batch_size'], self.props['latent_dim'])
                    ).to(device=self.props['device'])

                    # Training the generator

                    self.generator.zero_grad()

                    generated_samples = self.generator(latent_space_samples)
                    sample_noise = torch.rand((self.props['batch_size'], self.num_stations)).to(device=self.props['device'])

                    output_discriminator_generated = self.discriminator(
                        generated_samples + sample_noise * self.props['noise_scale']
                    ).to(device=self.props['device'])

                    loss_generator = self.props['loss_func'](
                        output_discriminator_generated, real_samples_labels
                    )

                    loss_generator.backward()

                    optimizer_generator.step()

                # Show loss

                if epoch % 10 == 0 and n == data_length // self.props['batch_size'] - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")

                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")

        os.makedirs(self.folder)
        torch.save(self.generator.state_dict(), os.path.join(self.folder, 'gen'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.folder, 'disc'))
        with open(os.path.join(self.folder, 'props.json'), 'w') as file:
            json.dump(self.props, file, cls=self.MyEncoder)

        print(f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

    def generate(self, num_samples):
        params_file = os.path.join(self.folder, 'gen')
        latent_variable = torch.rand(num_samples, self.props['latent_dim'])

        generator = Generator(self.props['latent_dim'], self.num_stations)
        generator.load_state_dict(torch.load(params_file))
        generator.eval()

        generated_samples = generator(latent_variable)
        generated_samples = generated_samples.detach().cpu().tolist()

        return np.asarray(generated_samples)

    class MyEncoder(JSONEncoder):
        def default(self, o):
            return str(o)
