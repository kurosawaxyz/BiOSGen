import torch
import torch.nn as nn
import clip 
import lora

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')

# Vision-Language Projector (MLP)
"""
VISION LANGUAGE PROJECTOR with MLP architecture
Idea inspired from MLP-Mixer: An all-MLP Architecture for Vision
Paper: https://arxiv.org/abs/2105.01601 by Tolstikhin et al. (2021)
"""
class VisionLanguageProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 256,
        device: str = "cuda",
        *args,
        **kwargs
    ):
        super(VisionLanguageProjector, self).__init__()
        