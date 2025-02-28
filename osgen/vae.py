import torch 
import torch.nn as nn
import torch.nn.functional as F

class VAEncoder(nn.module):
    def __init__(
        self, 
        input_dim: int = 256,
        output_dim: int = 128
    ):
        super(VAEncoder, self).__init__()
        