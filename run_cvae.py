from cvae import CVAE
from utils import *
import torch

name = input('Input the model name:')
cvae = CVAE(output_dim=1, latent_dim=4, conditional_dim=2, name=name)
data = load_data('df_full.csv')
stations = [[-3.242, -11.375],
            [-4.992, -0.425],
            [-0.292, 2.875],
            [7.758, 3.525],
            [1.608, -6.025],
            [-0.842, 11.425]]

train_set = [(torch.tensor([data[x, s]]), torch.tensor(stations[s])) for x in range(len(data)) for s in range(6)]
data_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

vae.fit(data_loader, 50)
