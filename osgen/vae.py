import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class VAEncoder(nn.Module):
    def __init__(self, input_dim=256, latent_dim=128):
        super(VAEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2_mean = nn.Linear(512, latent_dim)  # Mean for latent space
        self.fc2_logvar = nn.Linear(512, latent_dim)  # Log variance for latent space

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Sample noise
        return mean + eps * std  # Reparameterization trick

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=256):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Sigmoid to normalize output

# Define loss function

def vae_loss(recon_x, x, mean, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_divergence