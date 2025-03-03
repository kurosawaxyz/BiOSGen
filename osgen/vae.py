import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from abc import ABC, abstractmethod

import loralib as lora

class AbstractVAE(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

class VAEncoder(AbstractVAE):
    def __init__(
        self, 
        in_channels: int = 3,
        latent_channels: int = 4,
        out_channels_params: List[int] = [96, 128, 192,256],
        kernel_params: List[int] = [5, 3, 3, 3],
        pooling_layers: List[int] = [2, 4],
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
        if len(out_channels_params) != latent_channels:
            raise ValueError("Number of output channels must be equal to number of latent channels")
        if len(kernel_params) != latent_channels:
            raise ValueError("Number of kernel sizes must be equal to number of latent channels")

        # Initialize encoder layers
        for i in range(self.latent_channels):
            modules = []        # Initialize list of layers for each latent channel

            # First layer: Conv2d
            modules.append(lora.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels_params[i], 
                kernel_size=kernel_params[i], 
                stride=2, padding=2)
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
            kernel_size=3, 
            stride=1, padding=1
        )
        self.conv_logvar = lora.Conv2d(
            in_channels=out_channels_params[-1], 
            out_channels=latent_channels, 
            kernel_size=3, 
            stride=1, padding=1
        )

    def forward(self, x):
        for i in range(self.latent_channels):
            x = getattr(self, f"encoder_{i}")(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar
