import torch
import training
from training import device, plot_real, train, generator, plot_hist
import numpy as np
import os
import matplotlib.pyplot as plt



def run(model, params_file, output_length):
    latent_variable = torch.randn(
        output_length, model.latent_dimension
    )

    generator.load_state_dict(torch.load(params_file))
    generator.eval()

    generated_samples = generator(latent_variable)
    generated_samples = generated_samples.detach().cpu().tolist()
    
    return np.asarray(generated_samples)