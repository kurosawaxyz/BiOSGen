# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
import torch.nn.functional as F

from osgen.base import BaseModel

class StyleExtractor(BaseModel):
    def __init__(
        self,
        image_size: int = 512,
        embedded_dim: int = 64,
        activation: str = 'relu',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()

        self.activation = self._get_activation(activation)

        # Vanilla CNN encoder (no pretrained weights)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # (B, 64, H/2, W/2)
            nn.InstanceNorm2d(64, affine=True),
            self.activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 128, H/4, W/4)
            nn.InstanceNorm2d(128, affine=True),
            self.activation,

            nn.Conv2d(128, embedded_dim, kernel_size=3, stride=1, padding=1, bias=False),  # (B, embedded_dim, H/4, W/4)
            nn.InstanceNorm2d(embedded_dim, affine=True),
            self.activation,
        )

        # # Projection head per spatial location
        # self.fc_style = nn.Sequential(
        #     nn.Linear(embedded_dim, embedded_dim),
        #     self.activation,
        #     nn.Linear(embedded_dim, embedded_dim),
        # )

    def _get_activation(self, name):
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "elu": nn.ELU()
        }.get(name.lower(), nn.ReLU())

    def forward(self, image):
        B = image.size(0)
        x = self.encoder(image)         # (B, embedded_dim, H/4, W/4)
        # x = x.view(B, x.size(1), -1)    # (B, C, N)
        # x = x.permute(0, 2, 1)          # (B, N, C)
        # x = self.fc_style(x)            # (B, N, C)
        # x = x.permute(0, 2, 1)          # (B, C, N)
        return x

    

class PositionalEmbedding(BaseModel):
    pass