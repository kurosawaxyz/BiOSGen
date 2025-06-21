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

def total_variation_loss(img, weight=1.0):
    loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return weight * loss



def wasserstein_color_loss(style_image, generated_image, lambda_color=1.0):
    """
    Approximate Wasserstein distance for color distribution matching.
    Uses sorting-based approach that maintains differentiability.
    """
    if style_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=style_image.shape[2:], 
                                      mode="bilinear", align_corners=False)
    
    total_loss = 0
    
    # Process each color channel
    for c in range(style_image.shape[1]):
        style_channel = style_image[:, c, :, :].flatten()
        gen_channel = generated_image[:, c, :, :].flatten()
        
        # Sort the pixel values (this maintains gradients)
        style_sorted, _ = torch.sort(style_channel)
        gen_sorted, _ = torch.sort(gen_channel)
        
        # Compute 1-Wasserstein distance (L1 norm of sorted differences)
        wasserstein_dist = torch.mean(torch.abs(style_sorted - gen_sorted))
        total_loss += wasserstein_dist
    
    return lambda_color * total_loss

def moment_matching_loss(style_image, generated_image, moments=[1, 2, 3, 4], lambda_color=1.0):
    """
    Match statistical moments (mean, variance, skewness, kurtosis) of color distributions.
    Very efficient and differentiable.
    """
    if style_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=style_image.shape[2:], 
                                      mode="bilinear", align_corners=False)
    
    total_loss = 0
    
    for c in range(style_image.shape[1]):
        style_channel = style_image[:, c, :, :].flatten()
        gen_channel = generated_image[:, c, :, :].flatten()
        
        channel_loss = 0
        for moment in moments:
            # Compute centered moments
            if moment == 1:
                # Mean
                style_moment = torch.mean(style_channel)
                gen_moment = torch.mean(gen_channel)
            else:
                # Higher order moments
                style_mean = torch.mean(style_channel)
                gen_mean = torch.mean(gen_channel)
                
                style_moment = torch.mean((style_channel - style_mean) ** moment)
                gen_moment = torch.mean((gen_channel - gen_mean) ** moment)
            
            moment_diff = torch.abs(style_moment - gen_moment)
            channel_loss += moment_diff
        
        total_loss += channel_loss
    
    return lambda_color * total_loss

def lab_color_loss(style_image, generated_image, lambda_color=1.0):
    """
    Color matching in LAB color space for perceptually uniform color alignment.
    RGB to LAB conversion is differentiable.
    """
    def rgb_to_lab(rgb_tensor):
        """Differentiable RGB to LAB conversion"""
        # Normalize to [0, 1] if not already
        rgb = (rgb_tensor + 1) / 2  # Assuming input in [-1, 1]
        
        # RGB to XYZ conversion matrix (sRGB)
        rgb_to_xyz_matrix = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb_tensor.device, dtype=rgb_tensor.dtype)
        
        # Reshape for matrix multiplication
        B, C, H, W = rgb.shape
        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # Convert to XYZ
        xyz = torch.matmul(rgb_flat, rgb_to_xyz_matrix.T)
        
        # XYZ to LAB (simplified, approximate)
        # Note: This is a simplified version for gradient flow
        L = torch.pow(xyz[:, 1], 1/3) * 116 - 16
        a = 500 * (torch.pow(xyz[:, 0], 1/3) - torch.pow(xyz[:, 1], 1/3))
        b = 200 * (torch.pow(xyz[:, 1], 1/3) - torch.pow(xyz[:, 2], 1/3))
        
        lab = torch.stack([L, a, b], dim=1)
        return lab.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    if style_image.shape != generated_image.shape:
        generated_image = F.interpolate(generated_image, size=style_image.shape[2:], 
                                      mode="bilinear", align_corners=False)
    
    # Convert to LAB space
    style_lab = rgb_to_lab(style_image)
    gen_lab = rgb_to_lab(generated_image)
    
    # Compute moment matching in LAB space
    total_loss = 0
    for c in range(3):  # L, A, B channels
        style_channel = style_lab[:, c, :, :].flatten()
        gen_channel = gen_lab[:, c, :, :].flatten()
        
        # Match mean and variance in LAB space
        style_mean = torch.mean(style_channel)
        gen_mean = torch.mean(gen_channel)
        
        style_var = torch.var(style_channel)
        gen_var = torch.var(gen_channel)
        
        mean_loss = torch.abs(style_mean - gen_mean)
        var_loss = torch.abs(style_var - gen_var)
        
        total_loss += mean_loss + var_loss
    
    return lambda_color * total_loss

def adaptive_color_loss(style_image, generated_image, lambda_color=1.0):
    """
    Combines multiple color alignment approaches for robust matching.
    Uses weighted combination of moment matching and soft histogram.
    """
    # Compute individual losses
    moment_loss = moment_matching_loss(style_image, generated_image, lambda_color=1.0)
    wasserstein_loss = wasserstein_color_loss(style_image, generated_image, lambda_color=1.0)
    
    # Weighted combination
    total_loss = 0.6 * moment_loss + 0.4 * wasserstein_loss
    
    return lambda_color * total_loss

def color_alignment_loss(style_image, generated_image, method='wasserstein', lambda_color=1.0, **kwargs):
    """
    Unified interface for different color alignment methods.
    
    Args:
        method: 'wasserstein', 'moment_matching', 'lab', 'adaptive'
        lambda_color: Weight for the color loss
        **kwargs: Additional parameters for specific methods
    """
    if method == 'wasserstein':
        return wasserstein_color_loss(style_image, generated_image, lambda_color=lambda_color)
    elif method == 'moment_matching':
        return moment_matching_loss(style_image, generated_image, lambda_color=lambda_color, **kwargs)
    elif method == 'lab':
        return lab_color_loss(style_image, generated_image, lambda_color=lambda_color)
    elif method == 'adaptive':
        return adaptive_color_loss(style_image, generated_image, lambda_color=lambda_color)
    else:
        raise ValueError(f"Unknown method: {method}")