from torch import nn

class Generator(nn.Module):

    def __init__(self, latent_dimension):
        super().__init__()

        self.latent_dim = latent_dimension
        self.model = nn.Sequential(
            nn.Linear(latent_dimension, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        output = self.model(x)

        return output