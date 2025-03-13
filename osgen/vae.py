import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from abc import ABC, abstractmethod

import loralib as lora

from .nn import CrossAttentionStyleFusion

# ENCODER
class AbstractVAE(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def get_pool_indices(self) -> Dict[int, torch.Tensor]:
        """
        Returns the pooling indices from the last forward pass.
        
        Returns:
            Dictionary mapping layer indices to pooling indices tensors
        """
        return self.pool_indices
    
    def get_pool_sizes(self) -> Dict[int, torch.Size]:
        """
        Returns the pre-pooling tensor sizes from the last forward pass.
        Useful for unpooling operations.
        
        Returns:
            Dictionary mapping layer indices to pre-pooling tensor sizes
        """
        return self.pool_sizes
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VAEncoder(AbstractVAE):
    """
    Variational Autoencoder (VAE) Encoder, adjusted for LoRA, to be plugged into Unet model or Resnet model later
    With MaxPool return_indices support
    """
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
        is_trainable: bool = True,
        lora_rank: int = 8,
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
        self.pool_indices = {}  # Store pool indices for each pooling layer
        self.pool_sizes = {}    # Store sizes for unpooling operations
        self.is_trainable = is_trainable
        self.lora_rank = lora_rank

        # Check compatibility of hyperparameters
        if len(in_channels_params) != latent_channels:
            raise ValueError("Number of input channels must be equal to number of latent channels")
        if len(out_channels_params) != latent_channels:
            raise ValueError("Number of output channels must be equal to number of latent channels")
        if len(kernel_params) != latent_channels:
            raise ValueError("Number of kernel sizes must be equal to number of latent channels")

        # Initialize encoder layers - maintain original structure
        for i in range(self.latent_channels):
            
            # Apply LoRA with the specified rank
            # Create a lora Conv2d layer with specified rank
            conv = lora.Conv2d(
                in_channels=in_channels_params[i], 
                out_channels=out_channels_params[i], 
                kernel_size=kernel_params[i], 
                stride=stride_params[i], 
                padding=padding_params[i],
                r=lora_rank)  # Use the lora_rank parameter here
            
            conv_modules = []
            conv_modules.append(conv)
            
            # Activation function
            if self.activation_function == "relu":
                conv_modules.append(nn.ReLU())
            elif self.activation_function == "silu":
                conv_modules.append(nn.SiLU())
            elif self.activation_function == "gelu":
                conv_modules.append(nn.GELU())
            elif self.activation_function == "celu":
                conv_modules.append(nn.CELU())
            elif self.activation_function == "tanh":
                conv_modules.append(nn.Tanh())
            else:
                raise ValueError("Activation function not supported")
                
            setattr(self, f"conv_{i}", nn.Sequential(*conv_modules))
            
            # Pooling layers (if applicable)
            if i in pooling_layers:
                setattr(self, f"pool_{i}", nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            
            # Batch normalization (separate to maintain exact structure)
            setattr(self, f"bn_{i}", nn.BatchNorm2d(out_channels_params[i]))

        # Convolution to reduce to latent channels (with LoRA)
        self.conv_mu = lora.Conv2d(
            in_channels=out_channels_params[-1], 
            out_channels=latent_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            r=lora_rank  # Use the lora_rank parameter here
        )
        self.conv_logvar = lora.Conv2d(
            in_channels=out_channels_params[-1], 
            out_channels=latent_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            r=lora_rank  # Use the lora_rank parameter here
            )
        # Mark only LoRA parameters as trainable at the module level
        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')

    def get_mu_logvar(self, x):
        # Clear the pool indices dict each forward pass
        self.pool_indices = {}
        self.pool_sizes = {}
        
        # Process through each encoder block, replicating original flow
        for i in range(self.latent_channels):
            # Apply convolution + activation
            x = getattr(self, f"conv_{i}")(x)
            
            # Apply pooling if this is a pooling layer
            if i in self.pooling_layers:
                # Store the pre-pooling size
                self.pool_sizes[i] = x.size()
                # Apply pooling with indices
                x, indices = getattr(self, f"pool_{i}")(x)
                # Store the indices
                self.pool_indices[i] = indices
            
            # Apply batch normalization
            x = getattr(self, f"bn_{i}")(x)
            
        # Calculate mu and logvar (unchanged)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        # print(f"Mean shape: {mu.shape}, Log Variance shape: {logvar.shape}")
        
        return mu, logvar
    
    def forward(self, x):
        mu, logvar = self.get_mu_logvar(x)
        # Apply reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    

# CONDITIONED VAE
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
        is_trainable: bool = True,
        lora_rank: int = 8,
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
            device=device,
            is_trainable=is_trainable,
            lora_rank=lora_rank
        )
        self.conditioning = CrossAttentionStyleFusion(latent_channels=latent_channels, cond_dim=cond_dim, lora_rank=lora_rank, is_trainable=is_trainable)
        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')
        
    def forward(self, x, condition_embedding):
        # Apply reparameterization
        mu, logvar = self.encoder.get_mu_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # print(f"Z shape: {z.shape}, Condition shape: {condition_embedding.shape}")
        # Apply conditioning
        conditioned_z = self.conditioning(z, condition_embedding)
        # print(f"Conditioned Z shape: {conditioned_z.shape}")
        return conditioned_z
    


# DECODER
class VAEDecoder(AbstractVAE):
    def __init__(
        self, 
        in_channels: int = 4,
        out_channels: int = 3,
        hidden_dims: list = [64, 128, 256],
        activation_function: str = "silu",
        # cond_dim: int = 256,
        device: str = "cuda",
        is_trainable: bool = True,
        lora_rank: int = 8,
        *args,
        **kwargs
    ):
        super(VAEDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.device = device

        # First layer: Projection (from 4 channels to the first hidden dimension)
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dims[0]),
        ) 
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        current_dim = hidden_dims[0]
        
        for i in range(1, len(hidden_dims)):
            decoder_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                lora.Conv2d(current_dim, hidden_dims[i] // 2, kernel_size=3, padding=1, r=lora_rank),
                nn.BatchNorm2d(hidden_dims[i] // 2),
                nn.SiLU() if activation_function == "silu" else nn.ReLU(),
                
                lora.Conv2d(hidden_dims[i] // 2, hidden_dims[i] // 2, kernel_size=3, padding=1, r=lora_rank),
                nn.BatchNorm2d(hidden_dims[i] // 2),
                nn.SiLU() if activation_function == "silu" else nn.ReLU()
            )
            self.decoder_layers.append(decoder_block)
            current_dim = hidden_dims[i] // 2

        # Final output layer
        self.final_layer = nn.Sequential(
            lora.Conv2d(current_dim, out_channels, kernel_size=3, padding=1, r=lora_rank),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            lora.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, r=lora_rank),
            nn.Sigmoid(),
        )
        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')

    def forward(self, x):
        # print("decoding")
        # print(f"Input shape: {x.shape}")

        # Initial projection
        x = self.proj(x)
        # print(f"After projection: {x.shape}\n")
        
        # Decoder layers
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # print(f"After decoder layer {i}: {x.shape}")
        # print()

        # Final output layer
        output = self.final_layer(x)
        # print(f"Final output shape: {output.shape}")
        
        return output