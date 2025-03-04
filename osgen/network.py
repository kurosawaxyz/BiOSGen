import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from abc import ABC, abstractmethod

import loralib as lora

class AbstractAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

class CrossAttention(AbstractAttention):
    def __init__(self, latent_dim=4, cond_dim=256, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (latent_dim / heads) ** -0.5
        
        # For latent vector (z from VAE)
        self.to_q = lora.Linear(latent_dim, latent_dim)
        
        # For condition embedding
        self.to_k = lora.Linear(cond_dim, latent_dim)
        self.to_v = lora.Linear(cond_dim, latent_dim)
        
        self.to_out = lora.Linear(latent_dim, latent_dim)
        
    def forward(self, z, cond):
        # z shape: [batch, 4, H, W] - reshape to [batch, H*W, 4]
        batch, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(batch, h*w, c) 
        
        # Multi-head attention calculation
        q = self.to_q(z_flat)
        k = self.to_k(cond)
        v = self.to_v(cond)
        
        # Split into heads
        q = q.reshape(batch, -1, self.heads, c // self.heads).transpose(1, 2)
        k = k.reshape(batch, -1, self.heads, c // self.heads).transpose(1, 2)
        v = v.reshape(batch, -1, self.heads, c // self.heads).transpose(1, 2)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch, h*w, c)
        out = self.to_out(out)
        
        # Reshape back to spatial dimensions
        out = out.reshape(batch, h, w, c).permute(0, 3, 1, 2)
        
        return out
    
class CrossAttentionBlock(AbstractAttention):
    def __init__(self, latent_channels=4, cond_dim=256):
        super().__init__()
        self.norm = nn.GroupNorm(2, latent_channels)
        self.proj_q = lora.Conv2d(latent_channels, latent_channels, 1)
        self.proj_k = lora.Linear(cond_dim, latent_channels)
        self.proj_v = lora.Linear(cond_dim, latent_channels)
        self.proj_out = lora.Conv2d(latent_channels, latent_channels, 1)
        
    def forward(self, x, cond):
        # x: [B, 4, 8, 8], cond: [B, 256]
        B, C, H, W = x.shape
        
        # Normalize and get query
        h = self.norm(x)
        q = self.proj_q(h).view(B, C, -1)  # [B, 4, 64]
        
        # Get key and value from condition
        k = self.proj_k(cond).unsqueeze(-1)  # [B, 4, 1]
        v = self.proj_v(cond).unsqueeze(-1)  # [B, 4, 1]
        
        # Attention
        weight = torch.bmm(q.permute(0, 2, 1), k)  # [B, 64, 1]
        weight = F.softmax(weight, dim=1)
        h = torch.bmm(v, weight.permute(0, 2, 1)).view(B, C, H, W)  # [B, 4, 8, 8]

        return x + self.proj_out(h)