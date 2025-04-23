# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import loralib as lora
import math

from osgen.base import BaseModel
from osgen.utils import Utilities


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Class: TimestepBlock
class TimestepBlock(BaseModel, ABC):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        x = Utilities.convert_to_float32(x)
        emb = Utilities.convert_to_float32(emb)
        for layer in self:
            if isinstance(layer, TimestepBlock) or type(layer).__name__ in ["ResBlock", "CrossAttentionStyleFusion"]:
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    
# Class: Attention
class AbstractAttention(BaseModel, ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
class CrossAttentionStyleFusion(AbstractAttention):
    def __init__(
        self, 
        latent_channels=4, 
        cond_dim=256,
        is_trainable: bool = True,
        lora_rank: int = 16,
        *args,
        **kwargs
    ):
        super().__init__()
        self.norm = nn.GroupNorm(2, latent_channels)
        self.proj_q = lora.Conv2d(latent_channels, latent_channels, 1, r=lora_rank)
        self.proj_k = lora.Linear(cond_dim, latent_channels, r=lora_rank)
        self.proj_v = lora.Linear(cond_dim, latent_channels, r=lora_rank)
        self.proj_out = lora.Conv2d(latent_channels, latent_channels, 1, r=lora_rank)
        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')

        
    def forward(self, x, cond):
        # Convert inputs to float32
        x = Utilities.convert_to_float32(x)
        cond = Utilities.convert_to_float32(cond)
        # x: [B, 4, H, W], cond: [B, 256]
        B, C, H, W = x.shape
        
        # Normalize and get query
        h = self.norm(x)
        q = self.proj_q(h)                      # [B, 4, H, W]
        q = q.reshape(B, C, -1)                 # [B, 4, H*W]
        
        # Get key and value from condition
        k = self.proj_k(cond).unsqueeze(-1)     # [B, 4, 1]
        v = self.proj_v(cond).unsqueeze(-1)     # [B, 4, 1]
        
        # Attention
        weight = torch.bmm(q.permute(0, 2, 1), k)  # [B, H*W, 1]
        weight = F.softmax(weight, dim=1)
        
        # Apply attention and reshape back to spatial dimensions
        h = torch.bmm(v, weight.permute(0, 2, 1))  # [B, 4, H*W]
        h = h.reshape(B, C, H, W)                  # [B, 4, H, W]

        return x + self.proj_out(h)
    
# Class: Upsample
class Upsample(BaseModel):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool = True,
        out_channels: int = None,
        *args,
        **kwargs
    ):
        """
        Upsample layer that can optionally include a convolutional layer.
        Args:
            in_channels (int): Number of input channels.
            use_conv (bool): Whether to use a convolutional layer after upsampling.
            out_channels (int): Number of output channels. If None, defaults to in_channels.
        """
        super().__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        if use_conv:
            self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = Utilities.convert_to_float32(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
    
# Class: Downsample
class Downsample(BaseModel):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool = True,
        out_channels: int = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                in_channels, 
                self.out_channels, 
                kernel_size=3, 
                stride=stride,
                padding=1,
                )
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = Utilities.convert_to_float32(x)
        return self.op(x)
    

class ResBlock(BaseModel):
    def __init__(
        self,
        emb_channels: int,
        dropout: float,                 # Bottleneck dropout
        in_channels: int = 4,
        use_conv=False,
        out_channels: int = None,
        up=False,
        down=False,
        use_scale_shift_norm=False,
        device=None,
        *args,
        **kwargs
    ):
        # Define parameters
        super().__init__()
        
        # Ensure valid channel values
        self.in_channels = max(in_channels, 1)
        self.out_channels = max(out_channels or in_channels, 1)
        
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.updown = up or down
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Explicitly set device and dtype
        self.device = device or torch.device('cpu')

        # Define layers
        # Layer 1
        self.in_layers = nn.Sequential(
            nn.BatchNorm2d(
                self.in_channels, 
                device=self.device, 
            ),
            nn.ReLU(),
            lora.Conv2d(
                self.in_channels, 
                self.out_channels, 
                kernel_size=3, 
                padding=1,
                device=self.device,
            )  
        )

        # Middle layers
        if up:
            self.h_upd = Upsample(
                in_channels, 
                use_conv, 
                out_channels, 
                device=self.device, 
            )
            self.x_upd = Upsample(
                in_channels, 
                use_conv, 
                out_channels, 
                device=self.device, 
            )
        elif down:
            self.h_upd = Downsample(
                in_channels, 
                use_conv, 
                out_channels, 
                device=self.device, 
            )
            self.x_upd = Downsample(
                in_channels, 
                use_conv, 
                out_channels, 
                device=self.device, 
            )
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.ReLU(),
            lora.Linear(
                emb_channels, 
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                device=self.device,
            )
        )

        # Last layers
        self.out_layers = nn.Sequential(
            nn.BatchNorm2d(
                self.out_channels, 
                device=self.device, 
            ), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            lora.Conv2d(
                self.out_channels, 
                self.out_channels, 
                kernel_size=3, 
                padding=1,
                device=self.device,
            )
        )

        # Skip connection
        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = lora.Conv2d(
                self.in_channels, 
                self.out_channels, 
                kernel_size=1,
                device=self.device,
            )


    def forward(self, x, emb):
        x = Utilities.convert_to_float32(x)
        emb = Utilities.convert_to_float32(emb)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = out_rest(h)
        else:
            h = h + emb_out[:, :, None, None]
            h = self.out_layers(h)

        res = self.skip_connection(x) + h

        return res