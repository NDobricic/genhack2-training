import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np


# create a VAE model
class VAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, output_dim, latent_dim):
            super().__init__()

            self.latent_dim = latent_dim
            self.model = nn.Sequential(
                nn.Linear(output_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim * 2)
            )

        def forward(self, x):
            x = self.model(x)

            # split the output into mean and variance vectors
            mu, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]

            # return the latent vectors
            return mu, log_var

    class Decoder(nn.Module):
        def __init__(self, output_dim, latent_dim):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

        def forward(self, x):
            x = self.model(x)

            return x

    def __init__(self, output_dim, latent_dim, name):
        super().__init__()

        self.current_epoch = 0

        self.name = name

        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.encoder = self.Encoder(output_dim, latent_dim)
        self.decoder = self.Decoder(output_dim, latent_dim)

        # define the Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), betas=[0.5, 0.999])

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)  # sampling epsilon
        z = mu + log_var * epsilon  # reparameterization trick
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # encode the input
        mu, log_var = self.encode(x)

        # sample a latent vector from the latent space
        z = self.reparameterize(mu, torch.exp(0.5 * log_var))

        # decode the latent vector
        output = self.decode(z)

        # return the decoded output
        return output, mu, log_var

    def fit(self, data_loader, num_epochs):
        self.train()

        # define the loss function
        def loss_fn(recon_x, x, mu, log_var):
            # compute the reconstruction loss
            recon_loss = F.mse_loss(recon_x, x)

            # compute the regularization term
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # return the total loss
            return recon_loss + kl_loss * 0.001

        # def loss_fn(x, x_hat, mean, log_var):
        #     reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        #     KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        #
        #     return reproduction_loss + KLD

        # train the VAE model
        for epoch in range(num_epochs):
            for data in data_loader:
                # extract the input data
                x = data.float()

                # reset the gradients
                self.optimizer.zero_grad()

                # forward pass
                recon_x, mu, log_var = self(x)

                # compute the loss
                loss = loss_fn(recon_x, x, mu, log_var)

                # backpropagate the gradients
                loss.backward()

                # update the model weights
                self.optimizer.step()

            if (self.current_epoch + epoch) % 10 == 0:
                print(f'Epoch {self.current_epoch + epoch}/{self.current_epoch + num_epochs}')

        self.current_epoch += num_epochs
        print('Training finished')

    def save(self):
        # create the save path for the model
        file_path = f'saved\\{self.name}.params'

        # create a dictionary containing the model's parameters
        state_dict = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        # save the state to a file
        torch.save(state_dict, file_path)

    def resume(self, file_path):
        # load the state from a file
        state_dict = torch.load(file_path)

        # load the model's parameters
        self.load_state_dict(state_dict['model_state_dict'])

        # load the optimizer's parameters
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def generate_sample(self):
        # generate latent vector
        latent_vector = torch.randn(self.latent_dim)

        # decode the latent vector
        sample = self.decode(latent_vector)

        # return the generated sample
        return sample
