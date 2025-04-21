# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
import numpy as np

from osgen.base import BaseModel

class StyleExtractor(BaseModel):
    """
    Style extractor using Resnet50-based architecture.
    """
    def __init__(
        self,
        image_size: int = 512,
        activation: str = 'relu',
        use_pretrained: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ) -> None:
        super().__init__()

        # Init basic parameters
        self.image_size = image_size
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

        # Load ResNet50 model
        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = nn.Sequential(*self.resnet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        x = self.resnet(x)
        # x = self.activation(x)
        return x
