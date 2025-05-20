# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
from omegaconf import OmegaConf

# os.chdir("../")
from preprocess.dataloader import AntibodiesTree
from osgen.pipeline import *
from preprocess.patches_utils import PatchesUtilities
from osgen.utils import Utilities

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

    # Define tumors tree
    # SRC antibodies
    tree_src = AntibodiesTree(
        image_dir = os.path.join(args.data, args.original),
        mask_dir = os.path.join(args.data, 'tissue_masks', args.original),
        npz_dir = os.path.join(args.data, 'bbox_info', f'{args.original}_{args.style}', args.original)
    )

    # DST antibodies
    tree_dst = AntibodiesTree(
        image_dir = os.path.join(args.data, args.style),
        mask_dir = os.path.join(args.data, 'tissue_masks', args.style),
        npz_dir = os.path.join(args.data, 'bbox_info', f'{args.original}_{args.style}', args.style)
    )

    # Print
    print("Nb antibodies: ", tree_src.get_nb_antibodies())
    print("Nb antibodies: ", tree_dst.get_nb_antibodies())

    # Randomly select patches
    patches_src = PatchesUtilities.get_image_patches(
        image = np.array(Image.open(tree_src.antibodies[torch.randint(0, len(tree_src.antibodies), (1,)).item()])),
        tissue_mask = PatchesUtilities.get_tissue_mask(np.array(Image.open(tree_src.antibodies[torch.randint(0, len(tree_src.antibodies), (1,)).item()]))),
    )

    patches_dst = PatchesUtilities.get_image_patches(
        image = np.array(Image.open(tree_dst.antibodies[torch.randint(0, len(tree_dst.antibodies), (1,)).item()])),
        tissue_mask = PatchesUtilities.get_tissue_mask(np.array(Image.open(tree_dst.antibodies[torch.randint(0, len(tree_dst.antibodies), (1,)).item()]))),
    )

    # Define pipeline
    pipeline = OSGenPipeline()

    # Check trainable parameters
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # Check float32 parameters
    count = 0
    count_float32 = 0
    for name, module in pipeline.named_modules():
        for _, param in module.named_parameters(recurse=False):
            # print(f"{name}.{param_name}: dtype={param.dtype}")
            if param.dtype == torch.float32:
                count_float32 += 1
            count += 1

    print(f"float32 percentage: {count_float32 / count * 100}%")


    # Train the model
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

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Training loop
    # Put on training mode
    pipeline.train()
    requires_grad = True

    # Early stopping parameters
    early_stopping_patience = cfg.early_stopping_patience
    epochs_without_improvement = 0
    best_loss = float("inf")

    # Save losses
    losses = []
    content_losses = []
    style_losses = []

    # Switch to main training with bfloat16
    for epoch in tqdm(range(num_epochs)):
        # print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_losses = []
        avg_loss = []

        for i, (src, dst) in enumerate(zip(patches_src, patches_dst)):
            # Convert to tensors
            src_tensor = Utilities.convert_numpy_to_tensor(src).to(device)
            dst_tensor = Utilities.convert_numpy_to_tensor(dst).to(device)

            timesteps = torch.randint(0, 1000, (batch_size,), device=device)  # Random timesteps

            # forward pass
            decoded = pipeline(src_tensor, dst_tensor, timesteps)

            # Handle gradients issues
            requires_grad = decoded.requires_grad
            # print(requires_grad)
            if not requires_grad:
                break

            # Compute loss
            content_l, style_l, total_loss = pipeline.compute_loss(src_tensor, dst_tensor, decoded, lambda_content, lambda_style)

            # Break if total_loss is NaN
            if torch.isnan(total_loss):
                print(f"NaN loss at epoch {epoch+1}, batch {i+1}")
                break

            avg_loss.append(total_loss.item())

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # losses.append(total_loss.item())
        current_loss = np.mean(avg_loss)
        losses.append(current_loss)
        content_losses.append(content_l.item())
        style_losses.append(style_l.item())

        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            if num_epochs > 100: 
                print(f"Early stopping at epoch {epoch+1}")
                break 

        if verbose: 
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(patches_src)}, "
                    f"Content Loss: {content_l.item():.4f}, Style Loss: {style_l.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    main()