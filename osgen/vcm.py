import torch 
import torch.nn as nn
import clip 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
from PIL import Image

def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

# Vision-Language Projector (MLP)
class VisionLanguageProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):  # Matching CLIP text embedding dim
        super(VisionLanguageProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)  # Align with CLIP text embeddings
        )

    def forward(self, x):
        return self.mlp(x)
    
def extract_style_emb(
        image_path: str, 
        model: clip.models.ViT, 
        preprocess: clip.transforms.Preprocess,
        show: bool = False
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        plt.show()
    return style_embedding