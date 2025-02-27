import torch 
import torch.nn as nn
import loralib as lora

class VAEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(VAEncoder, self).__init__()
        self.encoder = nn.Sequential(
            lora.Linear(input_dim, 1024),
            nn.ReLU(),
            lora.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
class VAEDecoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            lora.Linear(input_dim, 1024),
            nn.ReLU(),
            lora.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)