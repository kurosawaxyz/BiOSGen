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
    Formula: L_content = sum((phi_j(y_test) - phi_j(y_ref))^2) / (C_j * H_j * W_j)
    
    Note: The most effective loss among 4 losses, works well with low lambda and alone.
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", 
        align_corners=False)

    content_layers = ["21"]  
    # Use ReLU4_2 layer (high-level features)
    # 17 -> Use ReLU4_1 instead of ReLU4_2 (less strict)
    # 12 -> Use ReLU3_4 (even more flexible)
    orig_features = extract_features(original_image, content_layers)
    gen_features = extract_features(generated_image, content_layers)
    
    # Get feature maps
    orig_feat = orig_features["21"]
    gen_feat = gen_features["21"]
    
    # Calculate dimensions for normalization
    B, C, H, W = orig_feat.shape
    
    # Calculate content loss with proper normalization as per formula
    # L_content = sum((generated - content)^2) / (C * H * W)
    loss = torch.sum((gen_feat - orig_feat) ** 2) / (C * H * W)
    
    return lambda_content * loss


# Style Loss
def gram_matrix(features):
    """
    Compute Gram matrix of feature maps
    Formula: G_j(x) = sum(phi_j(y_test) * phi_j(y_ref)) / (C_j * H_j * W_j)
    """
    B, C, H, W = features.shape
    
    # Reshape features to (B, C, H*W)
    features = features.view(B, C, H * W)
    
    # Compute gram matrix: G = F * F^T
    gram_matrices = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by the number of elements (C * H * W) as per the formula
    return gram_matrices / (C * H * W)

def style_loss(style_image, generated_image, style_layers=None, layer_weights=None, lambda_style=1.0):
    """
    Compute style loss between style reference image and generated image.
    Style loss is the squared distance (Frobenius norm) between gram matrices.
    
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
    
    # Default layers if none provided - using lower and mid-level features for style
    if style_layers is None:
        style_layers = ["0", "5", "10", "19", "28", "32"]  # Including more layers for better style capture
    
    # Default weights if none provided
    if layer_weights is None:
        layer_weights = [1.0] * len(style_layers)
    
    style_features = extract_features(style_image, style_layers)
    gen_features = extract_features(generated_image, style_layers)
    
    total_loss = 0
    for i, layer in enumerate(style_layers):
        weight = layer_weights[i]
        
        # Get feature maps for this layer
        style_feat = style_features[layer]
        gen_feat = gen_features[layer]
        
        # Compute Gram matrices
        G_style = gram_matrix(style_feat)
        G_gen = gram_matrix(gen_feat)
        
        # Calculate Frobenius norm (squared distance between matrices)
        # This is the squared distance between the two gram matrices
        layer_loss = torch.sum((G_gen - G_style) ** 2)
        
        total_loss += weight * layer_loss
    
    return lambda_style * total_loss

# def total_loss(original_image, generated_image, 
#                lambda_structure=2e-4, lambda_color=1e-4, 
#                lambda_content=1e-3, lambda_style=1e-8, verbose=False):
#     """
#     Compute the total loss - simplified version
#     """
#     if original_image.shape != generated_image.shape:
#         generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

#     # Calculate individual losses with proper error handling
#     structure_loss_val = structure_preservation_loss(original_image, generated_image, lambda_structure)
#     # color_loss_val = color_alignment_loss(original_image, generated_image, bins=64, lambda_color=lambda_color)
#     content_loss_val = content_loss(original_image, generated_image, lambda_content)
#     style_loss_val = style_loss(original_image, generated_image, lambda_style)
    
#     # Replace NaN or Inf values with zeros
#     structure_loss_val = torch.nan_to_num(structure_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
#     color_loss_val = torch.nan_to_num(color_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
#     content_loss_val = torch.nan_to_num(content_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
#     style_loss_val = torch.nan_to_num(style_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
    
#     if verbose:
#         print(f"Structure Loss: {structure_loss_val.item():.6f}")
#         print(f"Color Loss: {color_loss_val.item():.6f}")
#         print(f"Content Loss: {content_loss_val.item():.6f}")
#         print(f"Style Loss: {style_loss_val.item():.6f}")

#     # Sum all losses
#     total = structure_loss_val + color_loss_val + content_loss_val + style_loss_val
#     return total