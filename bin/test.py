# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Basic libraries
import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import time
# Set random seed for reproducibility
torch.manual_seed(0)
import argparse
import re

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Personalized modules
from preprocess.dataloader import AntibodiesTree
from preprocess.patches_utils import PatchesUtilities
from osgen.utils import Utilities
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
    parser.add_argument("--results", required=True, help="Path to the results directory")
    parser.add_argument("--data", required=True, help="Path to the data directory")

    args = parser.parse_args()

    # Load the configuration file
    cfg = OmegaConf.load(args.config)
    data_dir = args.data
    checkpoints_dir = args.checkpoints
    original_stain = args.original
    style_stain = args.style

    # Check if checkpoints ends with .pth
    if not checkpoints_dir.endswith(".pth"):
        raise ValueError("Please provide a valid torch model for evaluation.")

    # SRC antibodies
    if original_stain != style_stain:
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
    else: 
        tree_src = AntibodiesTree(
            image_dir = os.path.join(data_dir, original_stain),
            mask_dir = os.path.join(data_dir, "tissue_masks", original_stain),
            npz_dir = os.path.join(data_dir, "bbox_info", f"{original_stain}_NKX3", original_stain)
        )

        # DST antibodies
        tree_dst = AntibodiesTree(
            image_dir = os.path.join(data_dir, style_stain),
            mask_dir = os.path.join(data_dir, "tissue_masks", style_stain),
            npz_dir = os.path.join(data_dir, "bbox_info", f"{original_stain}_NKX3", style_stain)
        )

    # Print
    print("Nb antibodies: ", tree_src.get_nb_antibodies())
    print("Nb antibodies: ", tree_dst.get_nb_antibodies())


    # Initialize your pipeline
    pipeline = OSGenPipeline()

    # Count the number of parameters in the pipeline
    num_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the pipeline: {num_params}")

    # Load the checkpoint file (a state_dict)
    checkpoints = torch.load(checkpoints_dir)
    match = re.search(r'/checkpoints/([^/]+)/', checkpoints_dir)
    if match:
        timestamps = match.group(1)
        print(timestamps)  # Output: 20250609-200558
    else:
        timestamps = "unknown"
    print("Loading checkpoints from: ", checkpoints_dir)

    # Create directory for saving data
    os.makedirs(args.results, exist_ok=True)

    results_dir = f"{args.results}/{timestamps}"
    os.makedirs(results_dir, exist_ok=True)

    print("Results directory created at:", results_dir)


    # Check for compatibility between checkpoint keys and pipeline state_dict keys
    print("Checking compatibility between checkpoint keys and pipeline state_dict keys...")
    checkpoint_keys = set(checkpoints.keys())
    pipeline_keys = set(pipeline.state_dict().keys())

    print("Number of keys only in checkpoint:", len(checkpoint_keys - pipeline_keys))
    print("Number of keys only in pipeline:", len(pipeline_keys - checkpoint_keys))
    print("Number of common keys:", len(checkpoint_keys & pipeline_keys))

    # Check for shape mismatches in common keys
    print("\nShape mismatches:")
    for key in checkpoint_keys & pipeline_keys:
        if checkpoints[key].shape != pipeline.state_dict()[key].shape:
            print(f"{key}: checkpoint shape = {checkpoints[key].shape}, pipeline shape = {pipeline.state_dict()[key].shape}")

    # Load the state_dict into the pipeline
    try: 
        pipeline.load_state_dict(checkpoints, strict=False)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        return
    print("State_dict loaded successfully.")

    # Set the pipeline to evaluation mode
    pipeline.eval()

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate
    for i in tqdm(range(len(tree_src.antibodies))):
        idx_src = i
        patches_src = PatchesUtilities.get_image_patches_full(
            image = np.array(Image.open(tree_src.antibodies[idx_src])),
        )

        idx_dst = torch.randint(0, len(tree_dst.antibodies), (1,)).item()
        patches_dst = PatchesUtilities.get_image_patches_full(
            image = np.array(Image.open(tree_dst.antibodies[idx_dst])),
        )

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

        # Save the generated image
        match = re.search(r'([^/]+)\.png$', tree_src.antibodies[idx_src])
        name_src = match.group(1) if match else "unknown_src"

        generated_image = Image.fromarray(generated)
        generated_image.save(os.path.join(results_dir, f"generated_{name_src}_{timestamps}.png"))

if __name__ == "__main__":
    main()