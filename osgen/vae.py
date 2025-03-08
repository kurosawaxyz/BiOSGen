import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from abc import ABC, abstractmethod

import loralib as lora

from .nn import CrossAttentionStyleFusion

# ENCODER

class AbstractVAE(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass


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
        stride_params: List[int] = [1, 1, 1, 1],    # Modified this to switch between 32, 16 and 8
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

    def get_mu_logvar(self, x):
        for i in range(self.latent_channels):
            x = getattr(self, f"encoder_{i}")(x)
            
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        print(f"Mean shape: {mu.shape}, Log Variance shape: {logvar.shape}")
        
        return mu, logvar
    
    def forward(self,x):
        mu, logvar = self.get_mu_logvar(x)
        # Apply reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
       
        
class ConditionedVAEncoder(AbstractVAE):
    def __init__(
        self, 
        in_channels: int = 3,
        latent_channels: int = 4,
        in_channels_params: List[int] = [3, 96, 128, 192],
        out_channels_params: List[int] = [96, 128, 192, 256],
        kernel_params: List[int] = [3, 3, 3, 3],
        stride_params: List[int] = [1, 1, 1, 1],  
        padding_params: List[int] = [1, 1, 1, 1],  
        pooling_layers: List[int] = [0,2],  
        activation_function: str = "relu",
        device: str = "cuda",
        cond_dim: int = 256,
        *args,
        **kwargs
    ):
        super(ConditionedVAEncoder, self).__init__()
        self.encoder = VAEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            in_channels_params=in_channels_params,
            out_channels_params=out_channels_params,
            kernel_params=kernel_params,
            stride_params=stride_params,
            padding_params=padding_params,
            pooling_layers=pooling_layers,
            activation_function=activation_function,
            device=device
        )
        self.conditioning = CrossAttentionStyleFusion(latent_channels=latent_channels, cond_dim=cond_dim)
        
    def forward(self, x, condition_embedding):
        # Apply reparameterization
        mu, logvar = self.encoder.get_mu_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        print(f"Z shape: {z.shape}, Condition shape: {condition_embedding.shape}")
        # Apply conditioning
        conditioned_z = self.conditioning(z, condition_embedding)
        return conditioned_z
    


# DECODER
class VAEDecoder(nn.Module):
    """
    Variational Autoencoder (VAE) Decoder for transforming UNet output back to image space.
    Designed to work with an output shape of [batch_size, 4, 128, 128].
    """
    def __init__(
        self, 
        in_channels: int = 4,     # UNet output channels
        out_channels: int = 3,    # RGB image
        hidden_dims: List[int] = [64, 128, 256, 512],
        activation_function: str = "silu",
        use_condition: bool = True,
        cond_dim: int = 256,
        device: str = "cuda",
        *args,
        **kwargs
    ):
        super(VAEDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.use_condition = use_condition
        self.device = device
        
        # Condition embedding integration
        if use_condition:
            self.condition_attention = CrossAttentionStyleFusion(
                latent_channels=in_channels,
                cond_dim=cond_dim
            )
        
        # Initial projection - maintain spatial dimensions
        self.proj = nn.Sequential(
            lora.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU() if activation_function == "silu" else nn.ReLU()
        )
        
        # Decoder architecture
        self.decoder_layers = nn.ModuleList()
        
        # Build decoder architecture with upsampling
        current_dim = hidden_dims[0]
        
        # Add decoder blocks with upsampling
        for i in range(1, len(hidden_dims)):
            decoder_block = nn.Sequential(
                # Upsampling convolution
                nn.Upsample(scale_factor=2, mode="nearest"),
                lora.Conv2d(current_dim, hidden_dims[i] // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[i] // 2),
                nn.SiLU() if activation_function == "silu" else nn.ReLU(),
                
                # Regular convolution
                lora.Conv2d(hidden_dims[i] // 2, hidden_dims[i] // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[i] // 2),
                nn.SiLU() if activation_function == "silu" else nn.ReLU()
            )
            self.decoder_layers.append(decoder_block)
            current_dim = hidden_dims[i] // 2
        
        # Final output layer to get back to 3 channels (RGB image)
        self.final_layer = nn.Sequential(
            lora.Conv2d(current_dim, current_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_dim // 2),
            nn.SiLU() if activation_function == "silu" else nn.ReLU(),
            lora.Conv2d(current_dim // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1] range
        )

        
    def forward(self, x, condition=None):
        """
        Forward pass of the decoder
        Args:
            x: Tensor of shape [batch_size, 4, 128, 128] (UNet output)
            condition: Optional condition embedding of shape [batch_size, 256]
        Returns:
            Decoded image of shape [batch_size, 3, 512, 512]
        """
        # Apply condition using cross-attention if available
        print(f"Input shape: {x.shape}")
        if self.use_condition and condition is not None:
            x = self.condition_attention(x, condition)

        print(f"Conditioned shape: {x.shape}")
        
        # Initial projection
        x = self.proj(x)
        print(f"Projected shape: {x.shape}")
        
        # Apply decoder layers with upsampling
        for layer in self.decoder_layers:
            x = layer(x)
            print(f"Layer {layer} done")
        
        # Final layer to get RGB image
        output = self.final_layer(x)
        print("Done")
        
        return output