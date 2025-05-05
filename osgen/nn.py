


# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math
from flash_attn import flash_attn_func
import inspect

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
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.bfloat16) / half
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


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, style=None):
        for layer in self:
            if isinstance(layer, StyledResBlock):
                x = layer(x, emb, style)
            elif hasattr(layer, "forward") and len(inspect.signature(layer.forward).parameters) > 2:
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class SpatialAdaIN(BaseModel):
    """Improved Adaptive Instance Normalization for spatial style transfer."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.channel_reducer = None
        
    def forward(self, content, style):
        # Convert to bfloat16
        content = Utilities.convert_to_bfloat16(content)
        style = Utilities.convert_to_bfloat16(style)
        
        # Resize style spatially to match content
        spatial_resized = F.interpolate(
            style, 
            size=(content.shape[2], content.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Match channels if needed
        if spatial_resized.shape[1] != content.shape[1]:
            if (self.channel_reducer is None or 
                self.channel_reducer.in_channels != spatial_resized.shape[1] or
                self.channel_reducer.out_channels != content.shape[1]):
                
                self.channel_reducer = nn.Conv2d(
                    spatial_resized.shape[1], 
                    content.shape[1], 
                    kernel_size=1
                ).to(content.device).to(torch.bfloat16)
                
            style_features = self.channel_reducer(spatial_resized)
        else:
            style_features = spatial_resized
        
        # Instance normalization on content (without affine parameters)
        content_mean = content.mean(dim=(2, 3), keepdim=True)
        content_var = content.var(dim=(2, 3), keepdim=True) + self.eps
        content_normalized = (content - content_mean) / torch.sqrt(content_var)
        
        # Extract style statistics with improved stability
        style_mean = style_features.mean(dim=(2, 3), keepdim=True)
        style_var = style_features.var(dim=(2, 3), keepdim=True) + self.eps
        style_std = torch.sqrt(style_var)
        
        # Apply style
        return style_std * content_normalized + style_mean
    
    
# Class: Attention
class FlashSelfAttention(BaseModel, ABC):
    def __init__(self, z_dim=64, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = z_dim // heads
        
        # Ensure z_dim is divisible by heads
        assert z_dim % heads == 0, f"z_dim {z_dim} must be divisible by heads {heads}"
        
        self.to_q = nn.Linear(z_dim, z_dim, bias=False)
        self.to_k = nn.Linear(z_dim, z_dim, bias=False)
        self.to_v = nn.Linear(z_dim, z_dim, bias=False)
        self.out_proj = nn.Linear(z_dim, z_dim)
        
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, z, return_attention=False):
        # Store original dtype for later conversion back
        original_dtype = z.dtype
        
        B, C, H, W = z.shape
        
        # Flatten z
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)
        
        # Compute Q, K, V from the same input
        Q = self.to_q(z_flat)  # (B, HW, C)
        K = self.to_k(z_flat)  # (B, HW, C)
        V = self.to_v(z_flat)  # (B, HW, C)
        
        # Reshape for multi-head attention
        Q = Q.view(B, H * W, self.heads, self.head_dim)
        K = K.view(B, H * W, self.heads, self.head_dim)
        V = V.view(B, H * W, self.heads, self.head_dim)

        # If requesting attention weights, compute them before conversion to fp16/bf16
        attention_weights = None
        if return_attention:
            # We'll compute the attention weights separately on CPU to avoid memory issues
            Q_vis = Q.detach().cpu().float()
            K_vis = K.detach().cpu().float()
            
            # Compute attention scores for all heads
            Q_vis = Q_vis.permute(0, 2, 1, 3)  # (B, heads, HW, head_dim)
            K_vis = K_vis.permute(0, 2, 1, 3)  # (B, heads, HW, head_dim)
            
            # Compute attention scores
            attn_scores = torch.matmul(Q_vis, K_vis.transpose(-1, -2)) * self.scale  # (B, heads, HW, HW)
            attention_weights = torch.softmax(attn_scores, dim=-1)  # (B, heads, HW, HW)
        
        # Convert to fp16 or bf16 for flash attention
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
            
        Q = Q.to(dtype)
        K = K.to(dtype)
        V = V.to(dtype)
        
        # Use flash_attn for self-attention
        attn_output = flash_attn_func(
            Q, K, V,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False
        )  # (B, HW, heads, head_dim)
        
        # Convert back to original dtype
        attn_output = attn_output.to(original_dtype)
        
        # Reshape back
        attn_output = attn_output.reshape(B, H * W, C)
        out = self.out_proj(attn_output)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        if return_attention:
            return out, attention_weights
        return out

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
        x = Utilities.convert_to_bfloat16(x)
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
        x = Utilities.convert_to_bfloat16(x)
        return self.op(x)
    

class ResBlock(BaseModel):
    """Simplified ResBlock without style conditioning."""
    def __init__(
        self,
        emb_channels: int,
        dropout: float,
        in_channels: int = 4,
        use_conv=False,
        out_channels: int = None,
        up=False,
        down=False,
        use_scale_shift_norm=True,  # Better for timestep conditioning
        device=None,
        *args,
        **kwargs
    ):
        super().__init__()
        
        # Ensure valid channel values
        self.in_channels = max(in_channels, 1)
        self.out_channels = max(out_channels or in_channels, 1)
        
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.updown = up or down
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Explicitly set device
        self.device = device or torch.device('cpu')

        # Layer 1
        self.in_norm = nn.GroupNorm(8, self.in_channels, device=self.device)  # GroupNorm for better perf
        self.in_act = nn.SiLU()  # SiLU for better gradients
        self.in_conv = nn.Conv2d(
            self.in_channels, 
            self.out_channels, 
            kernel_size=3, 
            padding=1,
            device=self.device
        )

        # Up/down operations
        if up:
            self.h_upd = Upsample(in_channels, use_conv, out_channels, device=self.device)
            self.x_upd = Upsample(in_channels, use_conv, out_channels, device=self.device)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv, out_channels, device=self.device)
            self.x_upd = Downsample(in_channels, use_conv, out_channels, device=self.device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Timestep embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels, 
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                device=self.device
            )
        )
        
        # Output layers
        self.out_norm = nn.GroupNorm(8, self.out_channels, device=self.device)
        self.out_act = nn.SiLU()
        self.out_dropout = nn.Dropout(p=dropout)
        self.out_conv = nn.Conv2d(
            self.out_channels, 
            self.out_channels, 
            kernel_size=3, 
            padding=1,
            device=self.device
        )

        # Skip connection
        if self.out_channels == self.in_channels and not self.updown:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                self.in_channels, 
                self.out_channels, 
                kernel_size=1,
                device=self.device
            )

    def forward(self, x, emb):
        """Forward pass without style conditioning."""
        # Convert inputs to bfloat16
        x = Utilities.convert_to_bfloat16(x)
        emb = Utilities.convert_to_bfloat16(emb)
        
        # Process skip connection early if we're up/downsampling
        if self.updown:
            h = self.in_norm(x)
            h = self.in_act(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_conv(h)
            skip = self.skip_connection(x)
        else:
            skip = self.skip_connection(x)
            h = self.in_norm(x)
            h = self.in_act(h)
            h = self.in_conv(h)
        
        # Process timestep embedding
        emb_out = self.emb_layers(emb)
        
        # Apply conditioning
        if self.use_scale_shift_norm:
            # Split embedding into scale and shift components
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            
            # Apply normalization and conditioning
            h = self.out_norm(h)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = self.out_act(h)
        else:
            h = h + emb_out[:, :, None, None]
            h = self.out_norm(h)
            h = self.out_act(h)
        
        h = self.out_dropout(h)
        h = self.out_conv(h)
        
        # Add skip connection
        return skip + h


class StyledResBlock(BaseModel):
    """ResBlock with AdaIN before and after."""
    def __init__(
        self,
        emb_channels: int,
        dropout: float,
        in_channels: int = 4,
        use_conv=False,
        out_channels: int = None,
        up=False,
        down=False,
        use_scale_shift_norm=True,
        style_strength=1.0,  # Control style influence
        device=None,
        *args,
        **kwargs
    ):
        super().__init__()
        
        # Ensure valid channel values
        self.in_channels = max(in_channels, 1)
        self.out_channels = max(out_channels or in_channels, 1)
        self.style_strength = style_strength
        
        # AdaIN layers
        self.pre_adain = SpatialAdaIN(self.in_channels)
        self.post_adain = SpatialAdaIN(self.out_channels)
        
        # Core ResBlock
        self.resblock = ResBlock(
            emb_channels=emb_channels,
            dropout=dropout,
            in_channels=in_channels,
            use_conv=use_conv,
            out_channels=out_channels,
            up=up,
            down=down,
            use_scale_shift_norm=use_scale_shift_norm,
            device=device,
            *args,
            **kwargs
        )
        
    def forward(self, x, emb, style):
        """Apply style before and after the ResBlock."""
        # Convert all inputs to bfloat16
        x = Utilities.convert_to_bfloat16(x)
        emb = Utilities.convert_to_bfloat16(emb)
        style = Utilities.convert_to_bfloat16(style)
        
        # Weighted style application before ResBlock
        if self.style_strength > 0:
            x_styled = self.pre_adain(x, style)
            x = x * (1 - self.style_strength) + x_styled * self.style_strength
        
        # Apply ResBlock
        h = self.resblock(x, emb)
        
        # Weighted style application after ResBlock
        if self.style_strength > 0:
            h_styled = self.post_adain(h, style)
            h = h * (1 - self.style_strength) + h_styled * self.style_strength
            
        return h