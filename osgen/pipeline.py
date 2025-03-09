import torch
import torch.nn as nn
from typing import Tuple

from .unet import UNetModel
from .vae import VAEncoder, VAEDecoder

class StyleTransferPipeline(nn.Module):
    """
    Complete pipeline for style transfer on tumor staining images
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        out_channels: int = 3,
        cond_dim: int = 256,
        device: str = "cuda",
        unet_model_channels: int = 64,
        unet_num_res_blocks: int = 2,
        unet_channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        *args,
        **kwargs
    ):
        super(StyleTransferPipeline, self).__init__()
        self.device = device

        # VAE Encoder
        self.encoder = VAEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            cond_dim=cond_dim,
            device=device
        )

        # UNet Model
        self.unet = UNetModel(
            in_channels=latent_channels,
            out_channels=latent_channels,
            model_channels=unet_model_channels,
            num_res_blocks=unet_num_res_blocks,
            channel_mult=unet_channel_mult,
            dropout=0.1,
            image_size=128
        )

        # VAE Decoder
        self.decoder = VAEDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            hidden_dims=[64, 128, 256, 512],
            device=device
        )

    def forward(
        self, 
        x: torch.Tensor, 
        style_condition: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the pipeline
        Args:
            x: Input image tensor of shape [batch_size, 3, H, W]
            style_condition: Style embedding tensor of shape [batch_size, 256]
            timesteps: Timestep tensor for UNet
        Returns:
            Styled output image of shape [batch_size, 3, H, W]
        """
        # Encode input to latent space
        latent = self.encoder(x, style_condition)
        
        # Process with UNet
        unet_output = self.unet(latent, timesteps)
        
        # Decode back to image space
        output = self.decoder(unet_output, style_condition)
        
        return output