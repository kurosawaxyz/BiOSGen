import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from abc import ABC, abstractmethod

import loralib as lora
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')

from .network import CrossAttention

class AbstractVAE(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def reparameterization_trick(self, x):
        """
        Reparameterization trick to sample from latent space
        """
        mu, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class VAEncoder(AbstractVAE):
    """
    Variational Autoencoder (VAE) Encoder, adjusted for LoRA, to be plugged into Unet model or Resnet model later
    """
    def __init__(
        self, 
        in_channels: int = 3,
        latent_channels: int = 4,
        in_channels_params: List[int] = [3, 96, 128, 192],
        out_channels_params: List[int] = [96, 128, 192, 256],
        kernel_params: List[int] = [3, 3, 3, 3],
        stride_params: List[int] = [2, 1, 1, 1],  
        padding_params: List[int] = [1, 1, 1, 1],  
        pooling_layers: List[int] = [0,2],  
        activation_function: str = "relu",
        device: str = "cuda",
        *args,
        **kwargs
    ):
        # Initialize hyperparameters
        super(VAEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.out_channels_params = out_channels_params
        self.kernel_params = kernel_params
        self.pooling_layers = pooling_layers
        self.activation_function = activation_function
        self.device = device

        # Check compatibility of hyperparameters
        if len(in_channels_params) != latent_channels:
            raise ValueError("Number of input channels must be equal to number of latent channels")
        if len(out_channels_params) != latent_channels:
            raise ValueError("Number of output channels must be equal to number of latent channels")
        if len(kernel_params) != latent_channels:
            raise ValueError("Number of kernel sizes must be equal to number of latent channels")

        # Initialize encoder layers
        for i in range(self.latent_channels):
            modules = []        # Initialize list of layers for each latent channel

            # First layer: Conv2d
            modules.append(lora.Conv2d(
                in_channels=in_channels_params[i], 
                out_channels=out_channels_params[i], 
                kernel_size=kernel_params[i], 
                stride=stride_params[i], 
                padding=padding_params[i])
            )
            # Activation function
            if self.activation_function == "relu":
                modules.append(nn.ReLU())
            elif self.activation_function == "silu":
                modules.append(lora.SiLU())
            else:
                raise ValueError("Activation function not supported")
            # Pooling layer
            if i in pooling_layers:
                modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # Batch normalization
            modules.append(nn.BatchNorm2d(out_channels_params[i]))
            # Add layers to the model
            setattr(self, f"encoder_{i}", nn.Sequential(*modules))

        # Convolution to reduce to 4 latent channels (instead of FC layers)
        self.conv_mu = lora.Conv2d(
            in_channels=out_channels_params[-1], 
            out_channels=latent_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0  # Modified: No padding for 1x1 conv
        )
        self.conv_logvar = lora.Conv2d(
            in_channels=out_channels_params[-1], 
            out_channels=latent_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0  # Modified: No padding for 1x1 conv
        )

    def forward(self, x):
        for i in range(self.latent_channels):
            x = getattr(self, f"encoder_{i}")(x)
            
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        print(f"Mean shape: {mu.shape}, Log Variance shape: {logvar.shape}")
        
        return mu, logvar
        