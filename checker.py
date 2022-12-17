from model_cond import generative_model
from utils import plot_results, load_data
import numpy as np

real_data = load_data('df_full.csv')

num_samples = len(real_data)
noise = np.random.randn(num_samples, 50)
position = [-3.242, -11.375, -4.992, -0.425, -0.292, 2.875, 7.758, 3.525, 1.608, -6.025, -0.842, 11.425]
output = generative_model(noise, position)

print('Plotting results...')
plot_results(real_data, output, f'checking submission')
