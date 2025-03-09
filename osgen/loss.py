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