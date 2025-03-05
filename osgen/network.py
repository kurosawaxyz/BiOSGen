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
    
class QKVAttention(AbstractAttention):
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