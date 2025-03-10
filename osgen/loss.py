import torch
import torch.nn.functional as F

def structure_preservation_loss(original_image, generated_image, lambda_structure=1.0):
    """
    Compute a loss that encourages structural preservation between the original and generated images.
    This uses a combination of MSE for low-level features and edge similarity.
    """
    if original_image.shape != generated_image.shape:
        print("Original and generated images have different shapes. Resizing generated image to match original image.")
        generated_image_resized = F.interpolate(generated_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    # 1. MSE for pixel-level preservation
    mse_loss = F.mse_loss(generated_image_resized, original_image)
    
    # 2. Edge detection using Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float32, device=original_image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float32, device=original_image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    
    # Apply edge detection
    orig_edges_x = F.conv2d(original_image, sobel_x, padding=1, groups=3)
    orig_edges_y = F.conv2d(original_image, sobel_y, padding=1, groups=3)
    orig_edges = torch.sqrt(orig_edges_x**2 + orig_edges_y**2)
    
    gen_edges_x = F.conv2d(generated_image_resized, sobel_x, padding=1, groups=3)
    gen_edges_y = F.conv2d(generated_image_resized, sobel_y, padding=1, groups=3)
    gen_edges = torch.sqrt(gen_edges_x**2 + gen_edges_y**2)
    
    # Edge preservation loss
    edge_loss = F.mse_loss(gen_edges, orig_edges)
    
    # Total loss
    total_loss = mse_loss + edge_loss
    return lambda_structure * total_loss

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
