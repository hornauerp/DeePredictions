import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super().__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

        # # Define parameters as nn.Parameter objects
        # self.W_EE = nn.Parameter(torch.Tensor([0.5]))  # Excitatory to excitatory weight
        # self.W_EI = nn.Parameter(torch.Tensor([0.4]))  # Inhibitory to excitatory weight
        # self.W_IE = nn.Parameter(torch.Tensor([0.3]))  # Excitatory to inhibitory weight
        # self.W_II = nn.Parameter(torch.Tensor([0.2]))  # Inhibitory to inhibitory weight

        # self.tau_exc = nn.Parameter(torch.Tensor([1.0]))  # Excitatory time constant
        # self.tau_inh = nn.Parameter(torch.Tensor([1.0]))  # Inhibitory time constant

        # self.I_ext_exc = nn.Parameter(
        #     torch.Tensor([0.5])
        # )  # External input to excitatory population
        # self.I_ext_inh = nn.Parameter(
        #     torch.Tensor([0.2])
        # )  # External input to inhibitory population

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)

        # Define the ODE system
        # E, I = x.split(2, dim=1)  # Assuming x is a tensor of shape (batch_size, 4)
        # dE_dt = (
        #     -E + self.elu(self.W_EE * E - self.W_EI * I + self.I_ext_exc)
        # ) / self.tau_exc
        # dI_dt = (
        #     -I + self.elu(self.W_IE * E - self.W_II * I + self.I_ext_inh)
        # ) / self.tau_inh

        # # Concatenate the results and add the output from the neural network
        # dxdt = torch.cat((dE_dt, dI_dt), dim=1)
        # out += dxdt

        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super().__init__()
        self.rnn = nn.GRU(obs_dim, nhidden)
        self.hid2lat = nn.Linear(nhidden, 2 * latent_dim)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.hid2lat(h.squeeze(0)).chunk(2, -1)  # Return mean and logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class VAE(nn.Module):
    def __init__(self, ode_func, recognition_rnn, decoder, obs_dim=2, latent_dim=4):
        super().__init__()
        self.ode_func = ode_func
        self.recognition_rnn = recognition_rnn
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

    def forward(self, x):
        # Encode the initial condition
        z0_mean, z0_logvar = self.recognition_rnn(x)
        epsilon = torch.randn(z0_mean.size()).to(z0_mean)
        z0 = z0_mean + torch.exp(0.5 * z0_logvar) * epsilon

        # Solve the ODE
        z = odeint(self.ode_func, z0, torch.linspace(0, 1, x.size(1)).to(z0.device))
        
        # Decode the solution
        x_hat = self.decoder(z)
        print(x.shape)
        print(x_hat.shape)

        return x_hat, z0_mean, z0_logvar


def vae_loss(x, x_hat, z0_mean, z0_logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp())

    return recon_loss + kl_div


def train(model, data_loader, device, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x in data_loader:
            x = x[0].to(device)
            optimizer.zero_grad()

            # Forward pass
            x_hat, z0_mean, z0_logvar = model(x)

            # Compute loss
            loss = vae_loss(x, x_hat, z0_mean, z0_logvar)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}")
