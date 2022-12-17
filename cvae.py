import torch
import torch.nn as nn
import torch.nn.functional as F


# create a CVAE model
class CVAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, output_dim, latent_dim, conditional_dim):
            super().__init__()

            self.latent_dim = latent_dim
            self.model = nn.Sequential(
                nn.Linear(output_dim + conditional_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim * 2)
            )

        def forward(self, x, c):
            x = torch.cat([x, c], dim=-1)
            x = self.model(x)

            # split the output into mean and variance vectors
            mu, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]

            # return the latent vectors
            return mu, log_var

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
            x = torch.cat([x, c], dim=-1)
            x = self.model(x)

            return x

    def __init__(self, df_train, conditionals, output_dim, latent_dim, conditional_dim, name, device='cpu'):
        super().__init__()

        self.df_min = df_train.min()
        self.df_max = df_train.max()
        df_train = (df_train - self.df_min) / (self.df_max - self.df_min)
        train_set = [(torch.tensor([df_train[x, s]]), torch.tensor(conditionals[s]))
                     for x in range(len(df_train)) for s in range(6)]
        self.data_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

        self.current_epoch = 0

        self.name = name
        self.device = torch.device(device)

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.conditional_dim = conditional_dim

        self.encoder = self.Encoder(output_dim, latent_dim, conditional_dim).to(device=self.device)
        self.decoder = self.Decoder(output_dim, latent_dim, conditional_dim).to(device=self.device)

        # define the Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, betas=[0.5, 0.999])

    def encode(self, x, c):
        return self.encoder(x, c)

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)  # sampling epsilon
        z = mu + log_var * epsilon  # reparameterization trick
        return z

    def decode(self, x, c):
        return self.decoder(x, c)

    def forward(self, x, c):
        # encode the input
        mu, log_var = self.encode(x, c)

        # sample a latent vector from the latent space
        z = self.reparameterize(mu, torch.exp(0.5 * log_var))

        # decode the latent vector
        output = self.decode(z, c)

        # return the decoded output
        return output, mu, log_var

    def fit(self, num_epochs):
        self.train()

        # define the loss function
        def loss_fn(recon_x, x, mu, log_var):
            # compute the reconstruction loss
            recon_loss = F.mse_loss(recon_x, x)

            # compute the regularization term
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # return the total loss
            return recon_loss + kl_loss * 0.001

        # def loss_fn(x, x_hat, mean, log_var):
        #     reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        #     KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        #
        #     return reproduction_loss + KLD

        # train the VAE model
        for epoch in range(num_epochs):
            for x, c in self.data_loader:
                # extract the input data
                x = x.float().to(device=self.device)
                c = c.float().to(device=self.device)

                # reset the gradients
                self.optimizer.zero_grad()

                # forward pass
                recon_x, mu, log_var = self(x, c)

                # compute the loss
                loss = loss_fn(recon_x, x, mu, log_var)

                # backpropagate the gradients
                loss.backward()

                # update the model weights
                self.optimizer.step()

            if (self.current_epoch + epoch) % 10 == 0:
                print(f'Epoch {self.current_epoch + epoch}/{self.current_epoch + num_epochs} LOSS: {loss}')

        self.current_epoch += num_epochs
        print('Training finished')

    def save(self):
        # create the save path for the model
        file_path = f'saved\\{self.name}-e{self.current_epoch}.params'

        # create a dictionary containing the model's parameters
        state_dict = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        # save the state to a file
        torch.save(state_dict, file_path)

        print(f'Saved parameters to {file_path}')

    def resume(self, file_path):
        # load the state from a file
        state_dict = torch.load(file_path)

        # load the model's parameters
        self.load_state_dict(state_dict['model_state_dict'])

        # load the optimizer's parameters
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def generate_sample(self, condition):
        # generate latent vector
        latent_vector = torch.randn(self.latent_dim).to(device=self.device)
        condition = torch.tensor(condition).to(device=self.device)

        # decode the latent vector
        sample = self.decode(latent_vector, condition) * (self.df_max - self.df_min) + self.df_min

        # return the generated sample
        return sample
