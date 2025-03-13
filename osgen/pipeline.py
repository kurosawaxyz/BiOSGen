import torch
import torch.nn as nn
from typing import Tuple

from .unet import UNetModel
from .vae import ConditionedVAEncoder, VAEDecoder

import loralib as lora

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
        unet_model_channels: int = 32,
        unet_num_res_blocks: int = 2,
        unet_channel_mult: Tuple[int, ...] = (1, 2, 4),
        is_trainable: bool = True,
        lora_rank: int = 16,
        use_conv: bool = False,
        *args,
        **kwargs
    ):
        super(StyleTransferPipeline, self).__init__()
        self.device = device

        # VAE Encoder
        # Image size of encoder input is 128x128x3
        self.encoder = ConditionedVAEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            cond_dim=cond_dim,
            device=device,
            is_trainable=is_trainable,
            lora_rank=lora_rank
        )

        self.unet = UNetModel(
            out_channels=latent_channels,
            model_channels=unet_model_channels,
            num_res_blocks=unet_num_res_blocks,
            dropout=0.1,
            in_channels=latent_channels,
            image_size=32,              # Image size of encoder output is 32x32x3
            use_scale_shift_norm=True,
            resblock_updown=False,
            num_classes=None,
            channel_mult=unet_channel_mult,
            is_trainable=is_trainable,
            lora_rank=lora_rank,
            use_conv=use_conv
        )

        # VAE Decoder
        self.decoder = VAEDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            hidden_dims=[64, 128, 256],
            device=device,
            is_trainable=is_trainable,
            lora_rank=lora_rank
        )
        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')

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

        # print("\nVAE Encoder Output Shape:", latent.shape,"\n")
        
        # Process with UNet
        unet_output = self.unet(latent, timesteps)
        # print("\nUnet Output Shape:", unet_output.shape,"\n")
        
        # Decode back to image space
        output = self.decoder(unet_output)
        # print("\nOutput Shape:", output.shape,"\n")
        
        return output
    
    def count_trainable_params(self):
        print("Encoder Trainable Params:", self.encoder.count_trainable_params())
        print("UNet Trainable Params:", self.unet.count_trainable_params())
        print("Decoder Trainable Params:", self.decoder.count_trainable_params())
        return sum(p.numel() for p in self.parameters() if p.requires_grad)