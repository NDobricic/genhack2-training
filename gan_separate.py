import os
import time

import numpy as np

import gan

class GAN:
    def __init__(self, props, num_stations, save_folder):
        self.folder = save_folder
        self.num_stations = num_stations
        self.gans = [gan.GAN(props, 1, os.path.join(save_folder, str(x))) for x in range(num_stations)]

    def __str__(self):
        return f'GAN at folder {self.folder}'

    def train(self, data):
        print(f'Training {self}')
        start_time = time.time()
        for i in range(self.num_stations):
            self.gans[i].train(np.transpose(np.asarray([data[:, i]])))

        print(f"Total time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

    def generate(self, num_samples):
        generated_samples = np.zeros((num_samples, self.num_stations))
        print(self.gans[0].generate(num_samples)[:, 0])
        for i in range(self.num_stations):
            generated_samples[:, i] = self.gans[i].generate(num_samples)[:, 0]

        print(generated_samples)

        return generated_samples

