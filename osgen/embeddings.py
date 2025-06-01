# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, resnet50, VGG19_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


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

        # Activation function
        self.activation = getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.ReLU()

        self.device = device

        # ResNet50 up to layer4
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, H/32, W/32)

        # VGG19 partial
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if use_pretrained else None)
        self.vgg = create_feature_extractor(
            vgg.features,
            return_nodes={
                '4': 'relu1_1',
                '9': 'relu2_1',
                '18': 'relu3_1'
            }
        )

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Downsample VGG output to match ResNet size
        self.vgg_reduce = nn.Sequential(
            nn.Conv2d(256, image_size, kernel_size=1, bias=False),  # assuming relu3_1 gives 256 channels
            nn.InstanceNorm2d(image_size),
            self.activation,
        )

        # Reduce ResNet channel dimension
        self.resnet_reduce = nn.Conv2d(2048, image_size, kernel_size=1, bias=False)

        # Combine both and encode to style
        self.conv_style = nn.Sequential(
            nn.Conv2d(image_size * 2, embedded_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embedded_dim),
            self.activation,
            nn.Conv2d(embedded_dim, embedded_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embedded_dim),
            self.activation,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, image):
        # ResNet path
        x_r = self.resnet(image)              # (B, 2048, H/32, W/32)
        x_r = self.resnet_reduce(x_r)         # (B, image_size, H/32, W/32)

        # VGG path
        vgg_feats = self.vgg(image)
        x_v = vgg_feats['relu3_1']            # (B, 256, H/8, W/8)
        x_v = F.adaptive_avg_pool2d(x_v, x_r.shape[-2:])  # Downsample to H/32
        x_v = self.vgg_reduce(x_v)            # (B, image_size, H/32, W/32)

        # Fuse
        x = torch.cat([x_r, x_v], dim=1)      # (B, image_size*2, H/32, W/32)
        x = self.conv_style(x)                # (B, embedded_dim, H/32, W/32)

        # Upsample to match original input
        x = self.upsample(self.upsample(self.upsample(x)))  # → H, W

        return x  # (B, embedded_dim, H, W)

class PositionalEmbedding(BaseModel):
    pass