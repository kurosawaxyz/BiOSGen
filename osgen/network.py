import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from abc import ABC, abstractmethod

import loralib as lora

def cross_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class AbstractAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

class CrossAttentionABC(nn.Module):
    def __init__(self, latent_dim=4, cond_dim=256, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (latent_dim // heads) ** -0.5
        
        # For latent vector (z from VAE)
        self.to_q = nn.Linear(latent_dim, latent_dim)
        
        # For condition embedding
        self.to_k = nn.Linear(cond_dim, latent_dim)
        self.to_v = nn.Linear(cond_dim, latent_dim)
        
        self.to_out = nn.Linear(latent_dim, latent_dim)
        
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
    def __init__(
        self, 
        latent_channels: int = 4,
        cond_dim: int = 256,
        heads: int = 8,
        *args,
        **kwargs
    ):
        super(CrossAttentionBlock, self).__init__()
        
