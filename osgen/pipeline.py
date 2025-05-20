# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
from osgen.vae import VanillaEncoder, VanillaDecoder
from osgen.base import BaseModel
from osgen.nn import *
from osgen.unet import *
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
            
    )