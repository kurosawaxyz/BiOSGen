# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List

from osgen.base import BaseModel

class VanillaVAE(BaseModel):

    """
    Source code: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        # --- Build Encoder ---
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # Convolution to reduce to latent channels (with LoRA)
        self.conv_mu = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=latent_dim, 
            kernel_size=1, 
            stride=1, 
            padding=0,
        )
        self.conv_logvar = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=latent_dim, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            )
        # self.fc_mu = nn.Linear(hidden_dims[-1] * 16 * 16, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1] * 16 * 16, latent_dim)

        self.noise_predictor = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16 * 16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                    kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.conv_mu(result)
        log_var = self.conv_logvar(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 16, 16)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor, kernel_size=3) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # Apply smoothing (like a Gaussian blur)
        eps = F.avg_pool2d(eps, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        return eps * std + mu

    def reparameterize_learned(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Instead of random noise, learn noise shape
        eps = self.noise_predictor(mu)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Performs the forward pass of the VAE, including encoding,
        reparameterization, and decoding.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of tensors [reconstructed, input, mu, log_var]
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples



class VanillaEncoder(VanillaVAE):
    """
    Encoder part of the Vanilla VAE + reparameterization trick.
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 learned: bool = True,
                 **kwargs) -> None:
        super(VanillaEncoder, self).__init__(in_channels, latent_dim, hidden_dims, **kwargs)
        self.learned = learned

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encode(input)
        if self.learned:
            z = self.reparameterize_learned(mu, log_var)
        else:
            z = self.reparameterize(mu, log_var)
        return z
    
class VanillaDecoder(VanillaVAE):
    """
    Note: review later
    Decoder part of the Vanilla VAE.
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaDecoder, self).__init__(in_channels, latent_dim, hidden_dims, **kwargs)
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        return self.decode(input)