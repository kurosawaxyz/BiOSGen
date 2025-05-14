# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
import torch.nn.functional as F
import torchvision.models as models

# Structure loss
def structure_preservation_loss(original_image, generated_image, lambda_structure=1.0):
    """
    Structural preservation loss combining pixel-level MSE and Sobel edge similarity.
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="nearest")

    # Ensure float32
    original_image = original_image.float()
    generated_image = generated_image.float()

    # Pixel-level MSE loss
    mse_loss = F.mse_loss(original_image, generated_image)

    # Sobel filters (for 3-channel RGB)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=original_image.device).repeat(3, 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=original_image.device).repeat(3, 1, 1, 1)

    # Edge maps
    orig_edges = torch.sqrt(F.conv2d(original_image, sobel_x, padding=1, groups=3)**2 +
                            F.conv2d(original_image, sobel_y, padding=1, groups=3)**2 + 1e-8)
    
    gen_edges = torch.sqrt(F.conv2d(generated_image, sobel_x, padding=1, groups=3)**2 +
                           F.conv2d(generated_image, sobel_y, padding=1, groups=3)**2 + 1e-8)

    # Edge loss
    edge_loss = F.mse_loss(gen_edges, orig_edges)

    return lambda_structure * (mse_loss + edge_loss)

# Color Alignment Loss
def color_alignment_loss(original_image, generated_image, bins=64, lambda_color=1.0):
    """
    Compute the Color Alignment Loss (LCA) between the original and generated images.
    Simplified version with reduced bins and better numerical stability.
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # Ensure images are in the range [0, 1]
    original_img = original_image.clamp(0, 1)
    generated_img = generated_image.clamp(0, 1)

    # Compute histograms for each channel (reduced bins for efficiency)
    hist_orig = torch.histc(original_img, bins=bins, min=0, max=1).view(1, -1)
    hist_gen = torch.histc(generated_img, bins=bins, min=0, max=1).view(1, -1)

    # Normalize histograms with epsilon for stability
    eps = 1e-8
    hist_orig = hist_orig / (hist_orig.sum() + eps)
    hist_gen = hist_gen / (hist_gen.sum() + eps)

    # Compute the square root of histograms with epsilon for stability
    hist_orig_sqrt = torch.sqrt(hist_orig + eps)
    hist_gen_sqrt = torch.sqrt(hist_gen + eps)

    # Compute L2 distance between the square roots of the histograms
    color_loss = F.mse_loss(hist_orig_sqrt, hist_gen_sqrt)

    return lambda_color * color_loss

# Load pre-trained VGG model (only once)
vgg = None

def get_vgg_model():
    """
    Lazy loading of VGG model to avoid unnecessary initialization
    """
    global vgg
    if vgg is None:
        vgg = models.vgg19(pretrained=True).features.eval()
        # Freeze parameters to avoid unnecessary computation
        for param in vgg.parameters():
            param.requires_grad = False
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
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    content_layers = ["21"]  # Use ReLU4_2 layer (high-level features)
    orig_features = extract_features(original_image, content_layers)
    gen_features = extract_features(generated_image, content_layers)
    
    loss = F.mse_loss(gen_features["21"], orig_features["21"])
    
    return lambda_content * loss

def gram_matrix(features):
    """
    Compute Gram matrix of feature maps with improved numerical stability
    """
    B, C, H, W = features.shape
    features = features.view(C, H * W)
    G = torch.mm(features, features.t())
    # Normalize with epsilon for numerical stability
    return G / (C * H * W + 1e-8)

def style_loss(original_image, generated_image, lambda_style=1.0):
    """
    Compute style loss between original and generated images.
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # Reduced number of layers for efficiency
    style_layers = ["0", "5", "10", "19"]
    orig_features = extract_features(original_image, style_layers)
    gen_features = extract_features(generated_image, style_layers)

    loss = 0
    for layer in style_layers:
        G_orig = gram_matrix(orig_features[layer])
        G_gen = gram_matrix(gen_features[layer])
        loss += F.mse_loss(G_gen, G_orig)

    return lambda_style * loss

def total_loss(original_image, generated_image, 
               lambda_structure=2e-4, lambda_color=1e-4, 
               lambda_content=1e-3, lambda_style=1e-8, verbose=False):
    """
    Compute the total loss - simplified version
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # Calculate individual losses with proper error handling
    structure_loss_val = structure_preservation_loss(original_image, generated_image, lambda_structure)
    color_loss_val = color_alignment_loss(original_image, generated_image, bins=64, lambda_color=lambda_color)
    content_loss_val = content_loss(original_image, generated_image, lambda_content)
    style_loss_val = style_loss(original_image, generated_image, lambda_style)
    
    # Replace NaN or Inf values with zeros
    structure_loss_val = torch.nan_to_num(structure_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
    color_loss_val = torch.nan_to_num(color_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
    content_loss_val = torch.nan_to_num(content_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
    style_loss_val = torch.nan_to_num(style_loss_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    if verbose:
        print(f"Structure Loss: {structure_loss_val.item():.6f}")
        print(f"Color Loss: {color_loss_val.item():.6f}")
        print(f"Content Loss: {content_loss_val.item():.6f}")
        print(f"Style Loss: {style_loss_val.item():.6f}")

    # Sum all losses
    total = structure_loss_val + color_loss_val + content_loss_val + style_loss_val
    return total