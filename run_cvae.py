from cvae import CVAE
from utils import *
import torch

name = input('Input the model name:')
data = load_data('df_full.csv')
stations = [[-3.242, -11.375],
            [-4.992, -0.425],
            [-0.292, 2.875],
            [7.758, 3.525],
            [1.608, -6.025],
            [-0.842, 11.425]]

cvae = CVAE(df_train=data, conditionals=stations, output_dim=1,
            latent_dim=3, conditional_dim=2, name=name, device="cpu")
