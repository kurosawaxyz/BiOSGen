# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Structure loss
def structure_preservation_loss(original_image, generated_image, lambda_structure=1.0):
    """
    Structural preservation loss combining pixel-level MSE and Sobel edge similarity.

    Note: Actually the same definition as content loss, structure is merely equal to content, this one hits really hard with color too so not recommended. 
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
def differentiable_histogram(img, bins=64, min_val=0.0, max_val=1.0, sigma=0.01):
    """
    Differentiable histogram using soft assignment via Gaussian kernels.
    Assumes input `img` is in [0, 1] and of shape [B, C, H, W].
    """
    B, C, H, W = img.shape
    device = img.device

    # Flatten image to [B, C, N]
    img_flat = img.view(B, C, -1)  # [B, C, N]

    # Bin centers
    bin_centers = torch.linspace(min_val, max_val, steps=bins, device=device).view(1, 1, bins, 1)  # [1, 1, Bins, 1]

    # Expand image and compute soft binning using Gaussian kernel
    img_exp = img_flat.unsqueeze(2)  # [B, C, 1, N]
    bin_diff = (img_exp - bin_centers) ** 2  # [B, C, Bins, N]
    soft_bins = torch.exp(-bin_diff / (2 * sigma**2))  # Gaussian weighting

    # Normalize over pixels to get distribution
    hist = soft_bins.sum(dim=-1)  # [B, C, Bins]
    hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)  # normalize to sum to 1

    return hist  # shape: [B, C, bins]

def color_alignment_loss(original_image, generated_image, bins=64, lambda_color=1.0):
    """
    Note: This loss is not recommended, it actually breaks the entire output image instead of aligning colors.
    """
    if original_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # Clamp and normalize to [0, 1]
    original_img = original_image.clamp(0, 1)
    generated_img = generated_image.clamp(0, 1)

    # Compute differentiable histograms
    hist_orig = differentiable_histogram(original_img, bins=bins)
    hist_gen = differentiable_histogram(generated_img, bins=bins)

    # Compute Hellinger-like distance using square root
    hist_orig_sqrt = torch.sqrt(hist_orig + 1e-8)
    hist_gen_sqrt = torch.sqrt(hist_gen + 1e-8)

    # Compute mean squared error between histograms
    color_loss = F.mse_loss(hist_orig_sqrt, hist_gen_sqrt)

    return lambda_color * color_loss





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
    features = features.view(C, H * W)
    G = torch.mm(features, features.t())
    # Normalize with epsilon for numerical stability
    return G / (C * H * W + 1e-8)

def style_loss(original_image, generated_image, lambda_style=1.0):
    """
    Compute style loss between original and generated images.

    Note: This variant of Style loss actually maintain the structure, and also the original color, leading the style color enable to be transferred.
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