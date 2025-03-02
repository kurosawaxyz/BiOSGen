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
        image_path: str,
        show: bool = False,
        device: str = "cuda"
):
    """
    Extract style embedding from an image using CLIP model combined with proposed Vision-Language Projector

    ------------------------------------------------------------------------------
    Args:
        image_path (str): Path to the image
        show (bool): Whether to display the style embedding as an image
        device (str): Device to run the model on
    Output: 
        style_embedding (torch.Tensor): Style embedding of the image
    """
    model, preprocess = load_clip(device)
    
    style_image = Image.open(image_path).convert("RGB")
    style_tensor = preprocess(style_image).unsqueeze(0).to(device)
    # Extract features using CLIP image encoder
    with torch.no_grad():
        image_features = model.encode_image(style_tensor)

    # Normalize the image features (similar to text embeddings)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)


    # Pass through Vision-Language Projector
    vl_projector = VisionLanguageProjector(input_dim=512, output_dim=256).to(device)
    style_embedding = vl_projector(image_features)

    print("Transformed Style Embedding Shape:", style_embedding.shape)

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(style_embedding.cpu().detach().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig('assets/style_embedding.png')
        plt.show()
    return style_embedding
        