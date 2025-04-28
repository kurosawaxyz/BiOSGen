# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch as th
import torch.nn as nn

from osgen.base import BaseModel
from osgen.nn import * 
from osgen.utils import Utilities


class ConvBlock(BaseModel):
    """
    Basic convolutional block with batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DownBlock(BaseModel):
    """
    Downsampling block with residual connection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.adain = SpatialAdaIN(out_channels)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU()
        )
        
        self.downsample = nn.MaxPool2d(2)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb, style):
        # Process input
        h = self.conv1(x)
        h = self.conv2(h)
        
        # Apply time embedding
        t_emb = self.time_proj(t_emb)
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Apply style via AdaIN
        h = self.adain(h, style)
        
        # Skip connection
        h = h + self.skip_connection(x)
        
        # Store before downsampling for skip connection
        skip = h
        
        # Downsample
        h = self.downsample(h)
        h = self.dropout(h)
        
        return h, skip


class UpBlock(BaseModel):
    """
    Upsampling block with skip connections.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128, dropout=0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlock(in_channels + out_channels, out_channels)  # +out_channels for skip connection
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.adain = SpatialAdaIN(out_channels)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU()
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, skip, t_emb, style):
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process
        h = self.conv1(x)
        h = self.conv2(h)
        
        # Apply time embedding
        t_emb = self.time_proj(t_emb)
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Apply style via AdaIN
        h = self.adain(h, style)
        
        # Dropout
        h = self.dropout(h)
        
        return h


class UNetModel(BaseModel):
    """
    A simplified UNet model with time and style conditioning.
    """
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, style_channels=512, base_channels=64, channel_mults=(1, 2, 4, 8)):
        super().__init__()

        self.attn_proj = FlashSelfAttention(z_dim=base_channels, heads=8)
        
        # Initial processing
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        self.style_channels = style_channels
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        channels = [base_channels]
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            self.downs.append(DownBlock(
                channels[-1], 
                out_channels,
                time_emb_dim=time_emb_dim,
                dropout=0.1
            ))
            channels.append(out_channels)
        
        # Middle blocks
        self.middle_conv1 = ConvBlock(channels[-1], channels[-1])
        self.middle_adain = SpatialAdaIN(channels[-1])
        self.middle_conv2 = ConvBlock(channels[-1], channels[-1])
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            self.ups.append(UpBlock(
                channels[-1], 
                out_channels,
                time_emb_dim=time_emb_dim,
                dropout=0.1
            ))
            channels.append(out_channels)
        
        # Final processing
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x, timesteps, style):
        """
        Forward pass through the UNet.
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            style: Style tensor [B, style_channels, H, W]
            
        Returns:
            Output tensor of same shape as input
        """
        # Time embedding
        t_emb = timestep_embedding(timesteps, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        
        # Initial processing
        h = self.init_conv(x)
        
        # Store skip connections
        skips = []
        
        # Downsampling
        for down in self.downs:
            h, skip = down(h, t_emb, style)
            skips.append(skip)
        
        # Middle blocks
        h = self.middle_conv1(h)
        h = self.middle_adain(h, style)
        h = self.middle_conv2(h)
        
        # Upsampling
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb, style)
            h = self.attn_proj(h)
        
        # Final processing
        return self.final_conv(h)