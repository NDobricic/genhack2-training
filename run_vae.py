from vae import VAE
from utils import *
import torch

name = input('Input the model name:')
vae = VAE(output_dim=6, latent_dim=6, name=name)
data = torch.tensor(load_data('df_test.csv'))
data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

vae.fit(data_loader, 200)
