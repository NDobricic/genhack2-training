import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd
import numpy as np

# load the data, ignoring the first column
batch_size = 32
latent_size = 6
encoder_hidden_size = latent_size * 2
decoder_hidden_size = latent_size * 2

data = pd.read_csv("df_train.csv", usecols=["s1", "s2", "s3", "s4", "s5", "s6"])

# remove missing values
data = data.dropna()

# scale the data to have a mean of 0 and a standard deviation of 1
# data = (data - data.mean()) / data.std()
data = (data - data.min()) / (data.max() - data.min())
data = data
print(data.max())
print(data.min())

# convert the data to a NumPy array
data = data.to_numpy()

# print the preprocessed data
print(data)


# define the dataset
class TemperatureDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_percentage = 0.9
train_size = int(train_percentage * len(data))
train_data, val_data = data[:train_size], data[train_size:]  # split the data into training and validation sets

# define the dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, latent_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, latent_size * 2)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    z_mean, z_variance = x[:, 0:latent_size], x[:, latent_size:2 * latent_size]
    z_variance = F.softplus(z_variance)  # make sure it is non-negative because need to take log later
    return z_mean, z_variance


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# define the VAE
class VAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        # define the encoder
        self.encoder = Encoder(6, encoder_hidden_size, latent_size)

        # define the decoder
        self.decoder = Decoder(latent_size, decoder_hidden_size, 6)

    def forward(self, x):
        # pass the input through the encoder
        z_mean, z_logvar = self.encoder(x)

        # sample from the latent space
        z = self.reparameterize(z_mean, z_logvar)

        # pass the latent variable through the decoder
        x_recon = self.decoder(z)

        return x_recon, z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        # sample from the latent space
        eps = torch.randn(z_mean.shape)
        z = z_mean + torch.sqrt(z_logvar) * eps
        return z


# define the loss function
def vae_loss(x, x_recon, z_mean, z_logvar):
    # reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # regularization loss
    reg_loss = -0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - torch.exp(z_logvar))

    # total loss
    return recon_loss + reg_loss


# create the VAE
model = VAE(latent_size)

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the VAE
num_epochs = 1000

for epoch in range(num_epochs):
    for x in train_dataloader:
        x = x.float()

        # forward pass
        x_recon, z_mean, z_logvar = model(x)

        # compute the loss
        loss = vae_loss(x, x_recon, z_mean, z_logvar)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate the model on the validation set
    val_loss = 0
    with torch.no_grad():
        for x in val_dataloader:
            x = x.float()

            # forward pass
            x_recon, z_mean, z_logvar = model(x)

            # compute the loss
            val_loss += vae_loss(x, x_recon, z_mean, z_logvar).item()

    # print the epoch loss
    print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_dataloader)}")

def plot_results(real, fake, title):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(6):
        subplot = figure.add_subplot(2, 3, i + 1)
        #subplot.set_ylim([0, 1])
        subplot.set_title(f'Station {i + 1}')
        sb.histplot([x[i] for x in real], kde=True, stat="density")
        sb.histplot([x[i] for x in fake], kde=True, stat="density")

    figure.tight_layout()

    plt.show()

# use the VAE to generate samples
z = torch.randn(1000, latent_size)
x_recon = np.asarray(model.decoder(z).detach().cpu().tolist())
print(x_recon)

real_data = data[:1000, :]

print('Plotting results...')
# plot_results2(real_data, 'Real')
# plot_results2(generated_data, file)
plot_results(real_data, x_recon, 'lmao')