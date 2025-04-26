# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math
from flash_attn import flash_attn_func

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

class FlashStyleCrossAttention(AbstractAttention):
    def __init__(self, z_dim=64, style_dim=512, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = z_dim // heads
        
        # Ensure z_dim is divisible by heads
        assert z_dim % heads == 0, f"z_dim {z_dim} must be divisible by heads {heads}"
        
        self.to_q = nn.Linear(z_dim, z_dim, bias=False)
        self.to_k = nn.Linear(style_dim, z_dim, bias=False)
        self.to_v = nn.Linear(style_dim, z_dim, bias=False)
        self.out_proj = nn.Linear(z_dim, z_dim)
        
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, z, style, return_attention=False):
        # Store original dtype for later conversion back
        original_dtype = z.dtype
        
        B, C, H, W = z.shape
        B, C_style, H_style, W_style = style.shape
        
        # Flatten z
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)
        
        # Flatten style
        style_flat = style.permute(0, 2, 3, 1).reshape(B, H_style * W_style, C_style)  # (B, HW, C_style)
        
        # Compute Q, K, V
        Q = self.to_q(z_flat)      # (B, HW, C)
        K = self.to_k(style_flat)  # (B, H_style*W_style, C)
        V = self.to_v(style_flat)  # (B, H_style*W_style, C)
        
        # Reshape for multi-head attention
        Q = Q.view(B, H * W, self.heads, self.head_dim)
        K = K.view(B, H_style * W_style, self.heads, self.head_dim)
        V = V.view(B, H_style * W_style, self.heads, self.head_dim)

        # If requesting attention weights, compute them before conversion to fp16/bf16
        # This is separate from the main computation path
        attention_weights = None
        if return_attention:
            # We'll compute the attention weights separately on CPU to avoid memory issues
            # and to not interfere with the flash_attn_func path
            Q_vis = Q.detach().cpu().float()
            K_vis = K.detach().cpu().float()
            
            # Compute attention scores for all heads
            # (B, HW, heads, head_dim) x (B, N, heads, head_dim)T -> (B, heads, HW, N)
            Q_vis = Q_vis.permute(0, 2, 1, 3)  # (B, heads, HW, head_dim)
            K_vis = K_vis.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
            
            # Compute attention scores
            attn_scores = torch.matmul(Q_vis, K_vis.transpose(-1, -2)) * self.scale  # (B, heads, HW, N)
            attention_weights = torch.softmax(attn_scores, dim=-1)  # (B, heads, HW, N)
        
        # Convert to fp16 or bf16 for flash attention
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
            
        Q = Q.to(dtype)
        K = K.to(dtype)
        V = V.to(dtype)

        
        # Use flash_attn for cross-attention
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

# We don't use this class in the current code, but it's here for reference    
class CrossAttentionStyleFusion(AbstractAttention):
    def __init__(
        self, 
        latent_channels=4, 
        cond_dim=256,
        *args,
        **kwargs
    ):
        super().__init__()
        self.norm = nn.GroupNorm(2, latent_channels)
        self.proj_q = nn.Conv2d(latent_channels, latent_channels, 1)
        self.proj_k = nn.Linear(cond_dim, latent_channels)
        self.proj_v = nn.Linear(cond_dim, latent_channels)
        self.proj_out = nn.Conv2d(latent_channels, latent_channels, 1)

        
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
            nn.Conv2d(
                self.in_channels, 
                self.out_channels, 
                kernel_size=3, 
                padding=1,
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
            nn.Linear(
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
            nn.Conv2d(
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