import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

from omegaconf import OmegaConf
import argparse

# Import necessary modules (ensure these are in your project structure)
from osgen.pipeline import StyleTransferPipeline
from osgen.vit import extract_style_embedding
from osgen.dataloader import PatchDataLoader
from osgen.loss import total_loss, structure_preservation_loss, color_alignment_loss, style_loss, content_loss

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluation script for style transfer model")
parser.add_argument("--config_path", type=str, help="Configuration path", required=True)
parser.add_argument("--style_path", type=str, help="Style tumor path", required=True)
parser.add_argument("--original_path", type=str, help="Original tumor path", required=True)
parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint", required=True)
parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")

# Parse arguments
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load configuration
cfg = OmegaConf.load(args.config_path)
device = cfg.device

# Initialize model
model = StyleTransferPipeline(device=device)
model = model.to(device)

# Load checkpoint
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint, strict=False)

# Prepare data loader
tissue_mask_params = cfg.Image.tissue_mask_params
patch_extraction_params = cfg.Image.patch_extraction_params

data_loader = PatchDataLoader(
    path_src=args.original_path,
    path_dst=args.style_path,
    tissue_mask_params=tissue_mask_params,
    patch_extraction_params=patch_extraction_params,
    batch_size=3,
    val_ratio=0.15,
    test_ratio=0.15
)

# Evaluation metrics storage
eval_metrics = {
    'total_loss': [],
    'structure_loss': [],
    'content_loss': [],
    'style_loss': [],
    'color_alignment_loss': []
}

print("Total data for eval:", len(data_loader.val_dataset))

# Switch model to evaluation mode
model.eval()
# model.eval()    # double eval to prevent merge LoRA parameters to basic model parameters

# Disable gradient computation
with torch.no_grad():
    # Use test dataset for evaluation
    for idx in range(len(data_loader.val_dataset)):
        print(f"Evaluating sample {idx+1}...")
        # Get batch
        batch, style = data_loader.test_dataset[idx]
        batch = batch.unsqueeze(0).to(device)
        style = style.to(device)
        
        # Extract style embedding
        style_emb = extract_style_embedding(style[0].permute(1,2,0).numpy().astype(np.uint8), device=device)
        
        # Random timesteps
        timesteps = torch.randint(0, 1000, (4,)).to(device)
        
        # Generate stylized image
        out = model(batch, style_emb, timesteps)
        
        # Compute loss components
        structure_l = structure_preservation_loss(
            original_image=batch, 
            generated_image=out[0].unsqueeze(0),
            lambda_structure=cfg.losses.lambda_structure,
        )
        
        ca_l = color_alignment_loss(
            original_image=batch,
            generated_image=out[0].unsqueeze(0),
            lambda_color=cfg.losses.lambda_ca
        )
        
        content_l = content_loss(
            original_image=batch,
            generated_image=out[0].unsqueeze(0),
            lambda_content=cfg.losses.lambda_content
        )

        style_l = style_loss(
            original_image=batch,
            generated_image=out[0].unsqueeze(0),
            lambda_style=cfg.losses.lambda_style
        )
        
        total_l = total_loss(
            original_image=batch, 
            generated_image=out[0].unsqueeze(0),
            lambda_structure=cfg.losses.lambda_structure,
            lambda_content=cfg.losses.lambda_content,
            lambda_style=cfg.losses.lambda_style,
            lambda_color=cfg.losses.lambda_ca
        )
        
        # Store metrics
        eval_metrics['total_loss'].append(total_l.item())
        eval_metrics['structure_loss'].append(structure_l.item())
        eval_metrics['content_loss'].append(content_l.item())
        eval_metrics['style_loss'].append(style_l.item())
        eval_metrics['color_alignment_loss'].append(ca_l.item())
        
        # Save generated images for a few samples
        if idx < 5:  # Save first 5 generated images
            # Convert tensor to image
            # generated_img = out[0].permute(1,2,0).detach().numpy()
            # plt.imsave(os.path.join(args.output_dir, f'generated_image_{idx}.png'), generated_img)

            from PIL import Image
            generated_img = out[0].permute(1,2,0).detach().numpy().astype(np.uint8)
            fake_A_pil = Image.fromarray(generated_img)
            fake_A_pil.save(os.path.join(args.output_dir, f'generated_image_{idx}.png'))
            

    # Plot losses 
    # Create figure with GridSpec
    fig, ax = plt.subplots(3, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1, 2]})
    # Plot the first four graphs
    ax[0, 0].plot(eval_metrics['structure_loss'])
    ax[0, 0].set_title("Structure Loss")
    ax[0, 1].plot(eval_metrics['color_alignment_loss'])
    ax[0, 1].set_title("Color Alignment Loss")
    ax[1, 0].plot(eval_metrics['content_loss'])
    ax[1, 0].set_title("Content Loss")
    ax[1, 1].plot(eval_metrics['style_loss'])
    ax[1, 1].set_title("Style Loss")
    
    # Remove unused axes in the last row
    fig.delaxes(ax[2, 0])
    fig.delaxes(ax[2, 1])

    # Add last graph spanning both columns
    big_ax = fig.add_subplot(3, 1, 3)
    big_ax.plot(eval_metrics['total_loss'])             # losses is a list -> no need to convert to numpy array
    big_ax.hlines(np.mean(eval_metrics['total_loss']), 0, len(eval_metrics['total_loss']), colors='r', linestyles='--')
    big_ax.set_xlabel("Num_epochs")
    big_ax.set_ylabel("Loss")
    # big_ax.set_title("Total loss")

    # Adjust layout
    plt.tight_layout()
    plt.title('Eval Loss')
    # plt.show()
    plt.savefig(f"{args.output_dir}/losses_epoch.png")
    # plt.savefig(f"{output_dir}/losses_epoch.png")

# Command
# python -m bin.eval --config_path configs/train_config.yml --style_path demo/img/A6_TMA_15_02_IVB_NKX.png --original_path demo/img/A4_TMA_15_02_IVB_HE.png --checkpoint_path checkpoints/latest_osgen.pt --output_dir archive
