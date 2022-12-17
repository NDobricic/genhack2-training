#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, conditional_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + conditional_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, c):
        x = x.float()
        c = c.float()
        x = torch.cat([x, c], dim=-1)
        x = self.model(x)

        return x


# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise, position):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """

    # ---------------------
    decoder = Decoder(1, 3, 2)

    latent_variable = noise[:, :3]

    # load my parameters (of dimension 10 in this example).
    # <!> be sure that they are stored in the parameters/ directory <!>
    state_dict = torch.load(os.path.join("parameters", "decoder"))
    attributes = torch.load(os.path.join("parameters", "attributes"))

    # load the attributes
    df_min = attributes['df_min']
    df_max = attributes['df_max']
    decoder.load_state_dict(state_dict)
    decoder.eval()

    num_samples = len(noise)
    output = np.zeros((num_samples, 6))
    for i in range(num_samples):
        for s in range(6):
            lat = torch.tensor(latent_variable[i])
            pos = torch.tensor(position[s * 2:(s + 1)*2])
            output[i, s] = decoder.forward(lat, pos).detach().cpu() * (df_max - df_min) + df_min

    return output
