# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Basic libraries
import torch
import torch.nn as n
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from typing import List
import torch.optim as optim
from tqdm import tqdm
from omegaconf import OmegaConf
import time
# Set random seed for reproducibility
torch.manual_seed(0)
import argparse
import torch.optim as optim
import matplotlib.gridspec as gridspec
from torchinfo import summary
# from torchviz import make_dot

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Personalized modules
from preprocess.dataloader import AntibodiesTree
from preprocess.patches_utils import PatchesUtilities
from osgen.embeddings import StyleExtractor
from osgen.utils import Utilities
from osgen.vae import VanillaVAE,VanillaEncoder, VanillaDecoder
from osgen.base import BaseModel
from osgen.nn import *
from osgen.unet import *
from osgen.loss import *
from osgen.pipeline import *

def main():
    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument("--config", required=True, help="Path to the config file.")
    parser.add_argument("--original", required=True, help="Original stain style")
    parser.add_argument("--style", required=True, help="Styled stain style")
    parser.add_argument("--checkpoints", required=True, help="Path to the checkpoints directory")
    parser.add_argument("--data", required=True, help="Path to the data directory")

    args = parser.parse_args()

    # Load the configuration file
    cfg = OmegaConf.load(args.config)
    data_dir = args.data
    checkpoints_dir = args.checkpoints
    original_stain = args.original
    style_stain = args.style

    # SRC antibodies
    tree_src = AntibodiesTree(
        image_dir = os.path.join(data_dir, original_stain),
        mask_dir = os.path.join(data_dir, "tissue_masks", original_stain),
        npz_dir = os.path.join(data_dir, "bbox_info", f"{original_stain}_{style_stain}", original_stain)
    )

    # DST antibodies
    tree_dst = AntibodiesTree(
        image_dir = os.path.join(data_dir, style_stain),
        mask_dir = os.path.join(data_dir, "tissue_masks", style_stain),
        npz_dir = os.path.join(data_dir, "bbox_info", f"{original_stain}_{style_stain}", style_stain)
    )

    # Print
    print("Nb antibodies: ", tree_src.get_nb_antibodies())
    print("Nb antibodies: ", tree_dst.get_nb_antibodies())

    # Split train_tree and test tree
    train_idx_src, test_idx_src = Utilities.train_test_split_indices(tree_src.antibodies)
    train_idx_dst, test_idx_dst = Utilities.train_test_split_indices(tree_dst.antibodies)
    print("Train src: ", len(train_idx_src), "Test src: ", len(test_idx_src))
    print("Train dst: ", len(train_idx_dst), "Test dst: ", len(test_idx_dst))


    # Initialize your pipeline
    pipeline = OSGenPipeline()

    # Count the number of parameters in the pipeline
    num_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the pipeline: {num_params}")

    # Hyperparameters
    verbose = cfg.verbose
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    optimizer = optim.AdamW([p for p in pipeline.parameters() if p.requires_grad], 
                            lr=cfg.optimizer.params.lr, weight_decay=cfg.optimizer.params.weight_decay, eps=cfg.optimizer.params.eps)

    channels = cfg.channels
    height = cfg.height
    width = cfg.width
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_content = cfg.lambda_content
    lambda_style = cfg.lambda_style
    lambda_ca = cfg.lambda_ca
    ca_type = cfg.ca_type
    print(f"Color Alignment Method: {ca_type}")
    # lambda_tv = cfg.lambda_tv

    # Training loop
    # Put on training mode
    pipeline.train()
    requires_grad = True

    # Early stopping parameters
    early_stopping_patience = cfg.early_stopping_patience
    epochs_without_improvement = 0
    best_loss = float("inf")

    # # Convergence check parameters
    # convergence_patience = 5  # Number of epochs to check for convergence
    # convergence_threshold = 0.001  # Minimum relative change in loss to continue training
    # loss_history = []  # Keep track of recent losses

    # Save losses
    losses = []
    content_losses = []
    style_losses = []
    ca_losses = []
    # tv_losses = []

    # Add LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Get time
    # Create checkpoints_dir if it does not exist
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Create directory for saving checkpoints
    checkpoint_dir = f"{checkpoints_dir}/{timestamp}_{original_stain}_{style_stain}_{ca_type}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Checkpoint directory created at:", checkpoint_dir)

    # Get model summary
    print("Model summary:")
    input1 = torch.randn(batch_size, channels, height, width, device=device)
    input2 = torch.randn(batch_size, channels, height, width, device=device)
    input3 = torch.randint(0, 1000, (input1.size(0),), device=device)  # Random timesteps
    summa = summary(pipeline, input_data=(input1, input2, input3), device=device.type)
    # Save model summary to a file
    with open(f"{checkpoint_dir}/model_summary.txt", "w") as f:
        f.write(str(summa))
    # # Draw the model graph
    # output = pipeline(input1, input2, input3)
    # make_dot(output, params=dict(pipeline.named_parameters())).render(f"{checkpoint_dir}/model_graph", format="png")


    # Record memory stats
    torch.cuda.reset_peak_memory_stats()

    # Define start time
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
        avg_loss = []
        content_loss = []
        style_loss = []
        ca_loss = []

        # Assuming patches_src and patches_dst are lists or numpy arrays
        # num_samples = 4 # len(patches_src)
        # for j in range(batch_size):
            # Select random patches from src and dst
        idx_src = train_idx_src[torch.randint(0, len(train_idx_src), (1,)).item()]
        patches_src = PatchesUtilities.get_image_patches_full(
            image = np.array(Image.open(tree_src.antibodies[idx_src])),
            # tissue_mask=PatchesUtilities.get_tissue_mask(np.array(Image.open(tree_src.antibodies[idx_src])))
        )

        idx_dst = train_idx_dst[torch.randint(0, len(train_idx_dst), (1,)).item()]
        patches_dst = PatchesUtilities.get_image_patches_full(
            image = np.array(Image.open(tree_dst.antibodies[idx_dst])),
            # tissue_mask=PatchesUtilities.get_tissue_mask(np.array(Image.open(tree_dst.antibodies[idx_dst])))
        )
        for i, (src, dst) in enumerate(zip(patches_src, patches_dst)):
            if i == batch_size: 
                break
            # batch_src = patches_src[i:i + batch_size]
            # batch_dst = patches_dst[i:i + batch_size]

            # Convert to tensors
            src_tensor = Utilities.convert_numpy_to_tensor(src).to(device)
            dst_tensor = Utilities.convert_numpy_to_tensor(dst).to(device)

            # print(f"src_tensor shape: {src_tensor.shape}, dst_tensor shape: {dst_tensor.shape}")

            # Random timesteps for each item in the batch
            b_size = src_tensor.size(0)
            timesteps = torch.randint(0, 1000, (b_size,), device=device)

            # Forward pass
            decoded = pipeline(src_tensor, dst_tensor, timesteps)

            # Handle gradients issues
            if not decoded.requires_grad:
                break

            # Compute loss
            content_l, style_l, ca_l, total_loss = pipeline.compute_loss(src_tensor, dst_tensor, decoded, lambda_content, lambda_style, lambda_ca, ca_type)

            # Break if total_loss is NaN
            if torch.isnan(total_loss):
                print(f"NaN loss at epoch {epoch+1}, batch {i+1}")
                break

            # print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(patches_src)}, "
            #         f"Content Loss: {content_l.item():.4f}, Style Loss: {style_l.item():.4f}, Color Alignment Loss: {ca_l.item():.4f}, "
            #         f"Total Loss: {total_loss.item():.4f}")

            avg_loss.append(total_loss.item())
            content_loss.append(content_l.item())
            style_loss.append(style_l.item())
            ca_loss.append(ca_l.item())
            # tv_loss.append(tv_l.item())

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Record losses
        current_loss = np.mean(avg_loss)
        losses.append(current_loss)
        content_losses.append(np.mean(content_loss))
        style_losses.append(np.mean(style_loss))
        ca_losses.append(np.mean(ca_loss))

        # Step the LR scheduler
        scheduler.step(current_loss)

        # Early stopping (original implementation)
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if verbose: 
            if epoch % 25 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{batch_size}, "
                    f"Content Loss: {content_l.item():.4f}, Style Loss: {style_l.item():.4f}, "
                    f"Color Alignment Loss: {ca_l.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}")
                
                # Save the checkpoints
                torch.save(pipeline.state_dict(), f"{checkpoint_dir}/pipeline_epoch_{epoch+1}.pth")
                print(f"Model saved at {checkpoint_dir}/pipeline_epoch_{epoch+1}.pth")

                for param_group in optimizer.param_groups:
                    print(f"Current learning rate: {param_group['lr']}")

    # Define end time
    end_time = time.time()
    # Calculate total training time
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Shows current GPU memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Upper left: Content Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(content_losses, label='Content Loss')
    ax1.set_title('Content Loss', fontsize=15)
    ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Content Loss')

    # Upper right: Style Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(style_losses, label='Style Loss')
    ax2.set_title('Style Loss', fontsize=15)
    ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('Style Loss')

    # Upper middle: Color Alignment Loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ca_losses, label='Color Alignment Loss', color='orange')
    ax3.set_title('Color Alignment Loss', fontsize=15)
    ax3.set_xlabel('Epochs')

    # Bottom row (spanning both columns): Total Loss
    ax4 = fig.add_subplot(gs[1, :])  # spans all columns in row 2
    ax4.plot(losses, label='Total Loss', color='red')
    ax4.set_title('Total Loss', fontsize=15)
    ax4.set_xlabel(r'Epochs$\times$Batch Size')
    # ax4.set_ylabel('Total Loss')

    # Add title to fig
    fig.suptitle('Losses during Training', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for suptitle
    plt.savefig(f"{checkpoint_dir}/losses_plot.png")

    # Plot the last decoded, src_tensor and dst_tensor
    decoded = decoded.to(torch.uint8)
    src_tensor = src_tensor.to(torch.uint8)
    dst_tensor = dst_tensor.to(torch.uint8)
    fig, axes = plt.subplots(1, 6, figsize=(20, 15))  # 1x5 grid
    for i in range(3):
        ax = axes[i]
        ax.imshow(decoded[0, i].detach().cpu().numpy(), cmap='viridis')
        ax.set_title(f'Ch {i}', fontsize=8)
        # ax.axis('off')

    axes[3].imshow(decoded[0].permute(1,2,0).detach().cpu().numpy(), cmap='viridis')
    axes[3].set_title('Generated patch', fontsize=16)
    axes[4].imshow(src_tensor[0].permute(1,2,0).detach().cpu().numpy(), cmap='viridis')
    axes[4].set_title('Original patch', fontsize=16)
    axes[5].imshow(dst_tensor[0].permute(1,2,0).detach().cpu().numpy(), cmap='viridis')
    axes[5].set_title('Style patch', fontsize=16)
    plt.savefig(f"{checkpoint_dir}/decoded_src_dst.png")

    # After your training loop
    # Save model components
    torch.save(pipeline.state_dict(), f'{checkpoint_dir}/pipeline_best_{num_epochs}_{epoch+1}_epoch_512.pth')

    # Save the entire model state including optimizer
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': pipeline.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'losses': losses,
        'content_losses': content_losses,
        'style_losses': style_losses
    }

    torch.save(checkpoint, f'{checkpoint_dir}/pipeline_full_{num_epochs}_{epoch+1}_epoch_512.pth')

    # SAMPLE TEST GEN
    idx_src = 20 # torch.randint(0, len(tree_src.antibodies), (1,)).item()
    patches_src = PatchesUtilities.get_image_patches_full(
        image = np.array(Image.open(tree_src.antibodies[idx_src])),
    )

    idx_dst = 60 # torch.randint(0, len(tree_dst.antibodies), (1,)).item()
    patches_dst = PatchesUtilities.get_image_patches_full(
        image = np.array(Image.open(tree_dst.antibodies[idx_dst])),
    )
    print("idx_src: ", idx_src)
    print("idx_dst: ", idx_dst)

    fig, axes = plt.subplots(1, 3, figsize=(20, 15))  # 1x5 grid

    # Plot original source image
    axes[0].imshow(np.array(Image.open(tree_src.antibodies[idx_src])), cmap='viridis')
    axes[0].set_title(f'Original Tumor {idx_src}', fontsize=16)

    # Plot original destination image
    axes[1].imshow(np.array(Image.open(tree_dst.antibodies[idx_dst])), cmap='viridis')
    axes[1].set_title(f'Style Tumor {idx_dst}', fontsize=16)

    # Plot decoded image
    gen = []
    with torch.no_grad():
        for i in range(len(patches_src)):
            # Assign idx_src and idx_dst to the patches
            src = patches_src[i]
            dst = patches_dst[torch.randint(0, len(patches_dst), (1,)).item()]

            # Convert to tensors
            src_tensor = Utilities.convert_numpy_to_tensor(src).to(device)
            dst_tensor = Utilities.convert_numpy_to_tensor(dst).to(device)

            b_size = src_tensor.size(0)
            timesteps = torch.randint(0, 1000, (b_size,), device=device)  # Random timesteps

            # forward pass
            out = pipeline(src_tensor, dst_tensor, timesteps)
            gen.append(out.to(torch.uint8)[0].permute(1,2,0).detach().cpu().numpy())

    generated = PatchesUtilities.replace_patches_in_image_full(
        original_image=np.array(Image.open(tree_src.antibodies[idx_src])),
        generated_patches=gen
    )
    axes[2].imshow(generated, cmap='viridis')
    axes[2].set_title(f'Generated Image', fontsize=16)
    plt.savefig(f"{checkpoint_dir}/sample_test_gen.png")


    print(f"Training completed. Checkpoints saved in {checkpoint_dir}")

if __name__ == "__main__":
    main()