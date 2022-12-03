from torch import nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(6, 256),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(128, 64),

            nn.LeakyReLU(),

            nn.Dropout(0.4),

            nn.Linear(64, 6),

            nn.Sigmoid(),

        )

    def forward(self, x):
        output = self.model(x)

        return output