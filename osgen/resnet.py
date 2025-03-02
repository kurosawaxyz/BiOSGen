import torch
import torch.nn as nn

"""
Adaptation from https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/diffusionmodules/openaimodel.py
"""
class Resnet50LoRA(nn.Module):
    def __init__(
        self
    ):
        super(Resnet50LoRA, self).__init__()
        