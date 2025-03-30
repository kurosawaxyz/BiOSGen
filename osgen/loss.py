import torch
import torch.nn.functional as F
import torchvision.models as models

# Structure loss
def structure_preservation_loss(original_image, generated_image, lambda_structure=1.0):
    """
    Compute a loss that encourages structural preservation between the original and generated images.
    This uses a combination of MSE for low-level features and edge similarity.
    """
    if original_image.shape != generated_image.shape:
        # print("Original and generated images have different shapes. Resizing generated image to match original image.")
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)


    # 1. MSE for pixel-level preservation
    mse_loss = F.mse_loss(generated_image, original_image)
    
    # 2. Edge detection using Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float32, device=original_image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float32, device=original_image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    
    # Apply edge detection
    orig_edges_x = F.conv2d(original_image, sobel_x, padding=1, groups=3)
    orig_edges_y = F.conv2d(original_image, sobel_y, padding=1, groups=3)
    orig_edges = torch.sqrt(orig_edges_x**2 + orig_edges_y**2)
    
    gen_edges_x = F.conv2d(generated_image, sobel_x, padding=1, groups=3)
    gen_edges_y = F.conv2d(generated_image, sobel_y, padding=1, groups=3)
    gen_edges = torch.sqrt(gen_edges_x**2 + gen_edges_y**2)
    
    # Edge preservation loss
    edge_loss = F.mse_loss(gen_edges, orig_edges)
    
    # Total loss
    total_loss = mse_loss + edge_loss
    return lambda_structure * total_loss

# Color Alignment Loss
def color_alignment_loss(original_image, generated_image, bins=256, lambda_color=1.0):
    """
    Compute the Color Alignment Loss (LCA) between the original and generated images.
    The loss compares the square root of the color histograms of the two images.
    
    Args:
    - original_image: The target image (shape: [B, C, H, W]).
    - generated_image: The generated image (shape: [B, C, H, W]).
    - bins: The number of bins for the histogram (default is 256).
    - lambda_color: A weight factor for the color loss (default is 1.0).
    
    Returns:
    - The computed color alignment loss.
    """
    if original_image.shape != generated_image.shape:
        # print("Original and generated images have different shapes. Resizing generated image to match original image.")
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # Ensure images are in the range [0, 1]
    original_image = original_image.clamp(0, 1)
    generated_image = generated_image.clamp(0, 1)

    # Compute histograms for each channel
    hist_orig = torch.histc(original_image, bins=bins, min=0, max=1).view(1, -1)  # [1, bins]
    hist_gen = torch.histc(generated_image, bins=bins, min=0, max=1).view(1, -1)  # [1, bins]

    # Normalize histograms
    hist_orig /= hist_orig.sum()
    hist_gen /= hist_gen.sum()

    # Compute the square root of histograms
    hist_orig_sqrt = torch.sqrt(hist_orig)
    hist_gen_sqrt = torch.sqrt(hist_gen)

    # Compute L2 distance between the square roots of the histograms
    color_loss = F.mse_loss(hist_orig_sqrt, hist_gen_sqrt)

    return lambda_color * color_loss

# Content loss
# Load a pre-trained VGG model (only for feature extraction)
vgg = models.vgg19(pretrained=True).features.eval()

def extract_features(image, model, layers):
    """ Extracts features from specified layers of a CNN """
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def content_loss(original_image, generated_image, lambda_content=1.0):
    """
    Compute content loss between original and generated images using VGG-19.
    Ensures the generated image retains high-level structures from the original.
    """
    if original_image.shape != generated_image.shape:
        # print("Original and generated images have different shapes. Resizing generated image to match original image.")
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    content_layers = ["21"]  # Use ReLU4_2 layer (high-level features)
    orig_features = extract_features(original_image, vgg, content_layers)
    gen_features = extract_features(generated_image, vgg, content_layers)
    
    loss = F.mse_loss(gen_features["21"], orig_features["21"])
    
    return lambda_content * loss

# Style loss
def gram_matrix(features):
    """ Compute Gram matrix of feature maps """
    B, C, H, W = features.shape
    features = features.view(C, H * W)
    G = torch.mm(features, features.t())  # Compute Gram matrix
    return G.div(C * H * W)  # Normalize

def style_loss(original_image, generated_image, lambda_style=1.0):
    """
    Compute style loss between original and generated images.
    Uses Gram matrices to compare feature correlations.
    """
    if original_image.shape != generated_image.shape:
        # print("Original and generated images have different shapes. Resizing generated image to match original image.")
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    style_layers = ["0", "5", "10", "19", "28"]  # VGG19 ReLU layers for style
    orig_features = extract_features(original_image, vgg, style_layers)
    gen_features = extract_features(generated_image, vgg, style_layers)

    loss = 0
    for layer in style_layers:
        G_orig = gram_matrix(orig_features[layer])
        G_gen = gram_matrix(gen_features[layer])
        loss += F.mse_loss(G_gen, G_orig)

    return lambda_style * loss


# Total variation loss
def safe_loss(loss_value):
    return torch.nan_to_num(loss_value, nan=0.0, posinf=0.0, neginf=0.0)

def total_loss(original_image, generated_image, 
               lambda_structure=2*10**-4, lambda_color=10**-4, 
               lambda_content=10**-3, lambda_style=10**-8, image_size=256, verbose=False):
    """
    Compute the total loss by combining:
    - Structure Preservation Loss
    - Color Alignment Loss
    - Content Loss
    - Style Loss
    """
    if original_image.shape != generated_image.shape:
        # print("Original and generated images have different shapes. Resizing generated image to match original image.")
        generated_image = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # Size of original image: [B, C, H, W]
    # Size of generated image: [B, C, H, W] (may be different from original)
    structure_loss_value = safe_loss(structure_preservation_loss(original_image, generated_image, lambda_structure))
    color_loss_value = safe_loss(color_alignment_loss(original_image, generated_image, image_size, lambda_color))
    content_loss_value = safe_loss(content_loss(original_image, generated_image, lambda_content))
    style_loss_value = safe_loss(style_loss(original_image, generated_image, lambda_style))
    if verbose:
        print("Structure Loss:", structure_loss_value.item())
        print("Color Loss:", color_loss_value.item())
        print("Content Loss:", content_loss_value.item())
        print("Style Loss:", style_loss_value.item())

    total = structure_loss_value + color_loss_value + content_loss_value + style_loss_value
    return total
