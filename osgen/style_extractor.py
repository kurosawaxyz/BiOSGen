# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import loralib as lora

from osgen.base import BaseModel

class StyleExtractor(BaseModel):
    """
    StyleExtractor class for extracting style features from images using a pre-trained ResNet50 model.
    This class is designed to be used as a PyTorch model and can be easily integrated into a larger
    neural network architecture.
    Args:
        image_size (int): The size of the input images. Default is 512.
        activation (str): The activation function to use. Default is 'relu'.
        use_pretrained (bool): Whether to use a pre-trained ResNet50 model. Default is True.
        device (str): The device to run the model on. Default is 'cuda' if available, otherwise 'cpu'.
    """
    def __init__(
        self,
        image_size: int = 512,
        activation: str = 'relu',
        use_pretrained: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lora_rank: int = 16,
        is_trainable: bool = True,
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

        # Load ResNet50 up to layer4 (excluding avgpool and fc)
        resnet = resnet50(pretrained=use_pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # output: (B, 2048, H/32, W/32)

        # Conv2d layer to reduce channel depth
        self.conv2d = lora.Conv2d(2048, 256, kernel_size=1, r=lora_rank)

        # Dynamically determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            dummy_output = self.conv2d(self.resnet(dummy_input))
            flattened_dim = dummy_output.view(1, -1).shape[1]  # e.g. (1, C*H*W)

        self.flatten = nn.Flatten()
        self.dense1 = lora.Linear(flattened_dim, 256, r=lora_rank)
        self.dense2 = lora.Linear(256, 18 * 512, r=lora_rank)
        self.dropout = nn.Dropout(0.5)

        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')

    def forward(self, image):
        x = self.resnet(image)         # (B, 2048, H/32, W/32)
        x = self.conv2d(x)             # (B, 256, H/32, W/32)
        x = self.flatten(x)            # (B, flattened_dim)
        x = self.dense1(x)             # (B, 256)
        x = self.dropout(x)            # (B, 256)
        x = self.dense2(x)             # (B, 9216)
        x = x.view(-1, 18, 512)        # (B, 18, 512)
        return x