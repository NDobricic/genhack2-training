import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, output_dim, latent_dim, conditional_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(output_dim + conditional_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 2)
        )

    def forward(self, x, c):
        x = torch.cat([x, c], dim=-1)
        x = self.model(x)

        # split the output into mean and variance vectors
        mu, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]

        # return the latent vectors
        return mu, log_var
