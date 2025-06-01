# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Content Loss
# Load pre-trained VGG model (only once)
vgg = None

def get_vgg_model():
    """
    Lazy loading of VGG model to avoid unnecessary initialization
    """
    global vgg
    if vgg is None:
        weights = VGG19_Weights.IMAGENET1K_V1  # or VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features.eval()    # currently using pretrained weights
        # Freeze parameters to avoid unnecessary computation
        for param in vgg.parameters():
            param.requires_grad = False     # put as True may give nice insights but slow down the training
    return vgg

def extract_features(image, layers):
    """
    Extracts features from specified layers of the VGG model.
    Simplified version with explicit device handling.
    """
    model = get_vgg_model()
    device = image.device
    model = model.to(device)
    
    features = {}
    x = image
    
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x.clone()

    return features

def content_loss(original_image, generated_image, lambda_content=1.0):
    """
    Compute content loss between original and generated images using VGG-19.

    Note: The most effective loss among 4 losses, works well with low lambda and alone.
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", 
        align_corners=False)

    content_layers = ["21"]  # Use ReLU4_2 layer (high-level features)
    orig_features = extract_features(original_image, content_layers)
    gen_features = extract_features(generated_image, content_layers)
    
    loss = F.mse_loss(gen_features["21"], orig_features["21"])
    
    return lambda_content * loss


# Style Loss
def gram_matrix(features):
    """
    Compute Gram matrix of feature maps with improved numerical stability
    """
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)  # Keep batch dimension
    
    # Reshape preserving batch dimension
    gram_matrices = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize with epsilon for numerical stability
    return gram_matrices / (C * H * W + 1e-8)

def style_loss(style_image, generated_image, style_layers=None, layer_weights=None, lambda_style=1.0):
    """
    Compute style loss between style reference image and generated image.
    
    Args:
        style_image: The style reference image (the one whose style we want to capture)
        generated_image: The image being generated/optimized
        style_layers: List of VGG layers to extract features from
        layer_weights: Optional weights for each layer's contribution
        lambda_style: Overall weight for style loss
    """
    if style_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, 
                                       size=style_image.shape[2:], 
                                       mode="bilinear", 
                                       align_corners=False)
    
    # Default layers if none provided
    if style_layers is None:
        style_layers = ["0", "5", "10", "19", "28"]  # Including more layers for better style capture
    
    # Default weights if none provided
    if layer_weights is None:
        layer_weights = [1.0] * len(style_layers)
    
    style_features = extract_features(style_image, style_layers)
    gen_features = extract_features(generated_image, style_layers)
    
    loss = 0
    for i, layer in enumerate(style_layers):
        weight = layer_weights[i]
        G_style = gram_matrix(style_features[layer])
        G_gen = gram_matrix(gen_features[layer])
        loss += weight * F.mse_loss(G_gen, G_style)
    
    return lambda_style * loss