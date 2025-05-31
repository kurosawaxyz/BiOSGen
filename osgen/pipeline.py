# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
from typing import List

from osgen.embeddings import StyleExtractor
from osgen.vae import VanillaEncoder, VanillaDecoder
from osgen.base import BaseModel
from osgen.unet import AdaINUNet
from osgen.loss import *

class OSGenPipeline(BaseModel):
    """
    Main pipeline of the BiOSGen model.
    """
    def __init__(
            self,

            # Encoder parameters
            in_channels_encoder: int = 3,
            latent_dim_encoder: int = 64,
            hidden_dim_encoder: int = [32, 64],

            # U-Net parameters
            in_channels_unet: int = 64,  # Changed from 3 to 64 to match your input
            out_channels_unet: int = 3,
            time_emb_dim_unet: int = 128,
            base_channels_unet: int = 64,
            channel_mults_unet: tuple = (1, 2, 4, 8),  # Creates a model with 4 resolution levels
            num_res_blocks_unet: int = 2,            # 2 residual blocks per resolution
            attention_resolutions_unet: tuple = (4, 8),  # Add attention at 8×8 and 16×16 resolutions
            dropout_unet: float = 0.1,
            style_strength_unet: float = 1.0,

            # Decoder parameters
            in_channels_decoder: int = 64,
            latent_dim_decoder: int = 64,
            hidden_dims_decoder: List = None,
            output_channels_decoder: int = 3,

            # Style extractor parameters
            image_size: int = 512,
            embedded_dim: int = 64,
            activation: str = 'relu',

            # Other parameters
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            **kwargs
    ):
        super().__init__()

        # Encoder
        self.encoder = VanillaEncoder(
                    in_channels=in_channels_encoder,
                    latent_dim=latent_dim_encoder,
                    hidden_dims=hidden_dim_encoder
                ).to(device)
        
        # U-Net 
        self.unet = AdaINUNet(
                in_channels=in_channels_unet,  # Changed from 3 to 64 to match your input
                out_channels=out_channels_unet,
                time_emb_dim=time_emb_dim_unet,
                base_channels=base_channels_unet,
                channel_mults=channel_mults_unet,  # Creates a model with 4 resolution levels
                num_res_blocks=num_res_blocks_unet,            # 2 residual blocks per resolution
                attention_resolutions=attention_resolutions_unet,  # Add attention at 8×8 and 16×16 resolutions
                dropout=dropout_unet,
                style_strength=style_strength_unet,
                device=device
            ).to(device)
        
        # Decoder
        self.decoder = VanillaDecoder(
                    in_channels=in_channels_decoder,
                    latent_dim=latent_dim_decoder,
                    hidden_dims=hidden_dims_decoder,
                    output_channels=output_channels_decoder
                ).to(device)

        # Style Extractor
        self.style_extractor = StyleExtractor(
                    image_size=image_size,
                    embedded_dim=embedded_dim,
                    activation=activation,
                    device=device
                ).to(device)
        

    def forward(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the pipeline.
        Args:
            src_tensor (torch.Tensor): Source tensor.
            dst_tensor (torch.Tensor): Destination tensor.
            timesteps (torch.Tensor): Timesteps.
        Returns:
            torch.Tensor: Output tensor.
        """
        B, C, H, W = 1, 64, 128, 128

        # Extract style from style tumor
        style_emb = self.style_extractor(dst_tensor)
        # # print(f"Style Embedding Shape: {style_emb.shape}")  # [1, 64, 16384]
        # style_flat = style_emb.to(device='cuda')  # [1, 64, 16384]
        # style = style_flat.view(B, C, H, W)

        # Encode
        encoded = self.encoder(src_tensor) 

        # Pass through U-Net
        unet_out = self.unet(encoded, timesteps, style_emb)

        # Decode
        decoded = self.decoder(unet_out)

        return decoded
    
    # Compute loss
    def compute_loss(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor, generated_image: torch.Tensor, lambda_content: torch.float = 0.001, lambda_style: torch.float = 0.0001) -> torch.Tensor:
        """
        Compute the loss for the pipeline.
        Args:
            src_tensor (torch.Tensor): Original image tensor.
            dst_tensor (torch.Tensor): Style image tensor.
            generated_image (torch.Tensor): Generated image tensor.
        Returns:
            torch.Tensor: Computed loss.
        """
        # Content loss 
        content_l = lambda_content * content_loss(
                    original_image=src_tensor,
                    generated_image=generated_image,
                    lambda_content=1.0  # Using 1.0 here since we're scaling outside
                )
        
        style_l = lambda_style * style_loss(
                    style_image=dst_tensor,
                    generated_image=generated_image,
                    lambda_style=1.0  # Using 1.0 here since we're scaling outside
                )
        
        total_loss = content_l + style_l
        return content_l, style_l, total_loss