import torch
import torch.nn as nn
import clip 
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')

import loralib as lora

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
        is_trainable: bool = True,
        lora_rank: int = 8,
        *args,
        **kwargs
    ):
        super(VisionLanguageProjector, self).__init__()
        self.device = device

        self.mlp = nn.Sequential(
            lora.Linear(input_dim, input_dim * 2, r=lora_rank),
            nn.ReLU(),
            lora.Linear(input_dim * 2, output_dim, r=lora_rank),
        )
        # Train all parameters !
        if is_trainable:
            self.mlp.train()
        # if is_trainable:
        #     lora.mark_only_lora_as_trainable(self, bias='lora_only')

    def forward(self, x):
        x = self.mlp(x)
        return x
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Utils function
def load_clip(device: str = "cuda"):
    """
    Load CLIP model and pre-processing pipeline
    
    ------------------------------------------------------------------------------
    Args:
        device (str): Device to run the model on
    Output:
        model (torch.nn.Module): CLIP model
        preprocess (torchvision.transforms.Compose): Pre-processing pipeline
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def extract_style_embedding(
        image_array: torch.Tensor,
        show: bool = False,
        device: str = "cuda",
        savefig: bool = False
):
    """
    Extract style embedding from an image using CLIP model combined with proposed Vision-Language Projector

    ------------------------------------------------------------------------------
    Args:
        image_array (torch.Tensor): torch Tensor of shape (C, H, W) representing the image
        show (bool): Whether to display the style embedding as an image
        device (str): Device to run the model on
    Output: 
        style_embedding (torch.Tensor): Style embedding of the image
    """
    model, preprocess = load_clip(device)

    style_image = Image.fromarray(image_array).convert("RGB")
    style_tensor = preprocess(style_image).unsqueeze(0).to(device)
    # Extract features using CLIP image encoder
    with torch.no_grad():
        image_features = model.encode_image(style_tensor)

    # Normalize the image features (similar to text embeddings)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # print("Image Features Shape:", image_features.shape)


    # Pass through Vision-Language Projector
    vl_projector = VisionLanguageProjector(input_dim=512, output_dim=256).to(device)
    style_embedding = vl_projector(image_features)

    # print("Transformed Style Embedding Shape:", style_embedding.shape)

    if show:
        _, ax = plt.subplots(2, 1, figsize=(12,2))
        ax[0].imshow(image_features.cpu().detach().numpy(), cmap='viridis')
        ax[0].axis('off')
        ax[0].set_title('Before Projection')
        ax[1].imshow(style_embedding.cpu().detach().numpy(), cmap='viridis')
        ax[1].axis('off')
        ax[1].set_title('After Projection')

        if savefig:
            plt.savefig('assets/style_embedding.png')

        plt.show()
    return style_embedding
        

        
