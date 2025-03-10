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
        model_params = {
            'out_channels': 4,
            'model_channels': 32,
            'num_res_blocks': 2,
            'dropout': 0.1,
            'in_channels': 4,
            'image_size': 32,
            'use_scale_shift_norm': True,
            'resblock_updown': False,  # Disable excessive downsampling
            'num_classes': None,
            'channel_mult': (1, 2, 4),  # Reduce max depth
            # 'device': torch.device('cpu'),
            # 'dtype': torch.float32
        }

        self.unet = UNetModel(**model_params)

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
        latent = self.encoder(x)

        print("\nVAE Encoder Output Shape:", latent.shape,"\n")
        
        # Process with UNet
        unet_output = self.unet(latent, timesteps)
        print("\nUnet Output Shape:", unet_output.shape,"\n")
        
        # Decode back to image space
        output = self.decoder(unet_output)
        print("\nOutput Shape:", output.shape,"\n")
        
        return output