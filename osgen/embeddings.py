# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
from torchvision.models import resnet50

from osgen.base import BaseModel

class StyleExtractor(BaseModel):
    def __init__(
        self,
        image_size: int = 512,
        embedded_dim: int = 64,
        activation: str = 'relu',
        use_pretrained: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ) -> None:
        super().__init__()

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.device = device

        # Load ResNet50 up to layer4
        resnet = resnet50(pretrained=use_pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # output: (B, 2048, H/32, W/32)

        # Conv layer to reduce channels
        self.conv_reduce = nn.Conv2d(2048, image_size, kernel_size=1, bias=False)  # (B, 512, H/32, W/32)

        self.conv_style = nn.Sequential(
            nn.Conv2d(image_size, embedded_dim, kernel_size=3, padding=1, bias=False),   # keep size
            nn.BatchNorm2d(embedded_dim),
            self.activation,
            nn.Conv2d(embedded_dim, embedded_dim, kernel_size=3, padding=1, bias=False),   # keep size
            nn.BatchNorm2d(embedded_dim),
            self.activation,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fc_style = nn.Sequential(
            nn.Linear(embedded_dim, embedded_dim),
            self.activation,
            nn.Linear(embedded_dim, embedded_dim),
        )


    def forward(self, image):
        x = self.resnet(image)       # (B, 2048, H/32, W/32)
        x = self.conv_reduce(x)      # (B, 512, H/32, W/32)
        x = self.conv_style(x)       # (B, embedded_dim, H/32, W/32)
        x = self.upsample(x)  # Double the size
        x = self.upsample(x)  # Double the size
        x = self.upsample(x)  # Double the size

        # Flatten and apply fc
        x = x.view(x.size(0), x.size(1), -1)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.fc_style(x)
        x = x.permute(0, 2, 1)
        return x  # structured feature map
    

class PositionalEmbedding(BaseModel):
    pass