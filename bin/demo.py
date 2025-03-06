# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import torch 
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
import clip 

from PIL import Image

import math
from typing import Tuple, List
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM 

from preprocess.tissue_mask import GaussianTissueMask
from preprocess.utils import *
from osgen.nn import *
from osgen.unet import UNetModel
from osgen.vae import VAEncoder, ConditionedVAEncoder
from osgen.vit import extract_style_embedding


# Define hyperparameters
Image.MAX_IMAGE_PIXELS = 100000000000

if __name__ == '__main__':
    print("Hello, World!")

    IMAGE_PATH = "/Users/hoangthuyduongvu/Desktop/BiOSGen/data/HE/A3_TMA_15_02_IVB_HE.png"
    STYLE_PATH = "/Users/hoangthuyduongvu/Desktop/BiOSGen/data/NKX3/A6_TMA_15_02_IIB_NKX.png"

    # tissue mask params
    tissue_mask_params = {
        'kernel_size': 20,
        'sigma': 20,
        'downsample': 8,
        'background_gray_threshold': 220
    }

    # patch extraction params
    patch_extraction_params = {
        'patch_size': 128,
        'patch_tissue_threshold': 0.7,
        'is_visualize': False
    }

    # Test preprocess
    image_src = read_image(IMAGE_PATH)
    image_dst = read_image(STYLE_PATH)

    tissue_mask_src = get_tissue_mask(image=image_src, **tissue_mask_params)
    tissue_mask_dst = get_tissue_mask(image=image_dst, **tissue_mask_params)

    # extract src patches
    patches_src = get_image_patches(
        image=image_src,
        tissue_mask=tissue_mask_src,
        **patch_extraction_params
    )
    # extract dst patches
    patches_dst = get_image_patches(
        image=image_dst,
        tissue_mask=tissue_mask_dst,
        **patch_extraction_params
    )
    patch_src = normalize_patch(patches_src[100])
    patch_dst = normalize_patch(patches_dst[65])

    style_embedding = extract_style_embedding(STYLE_PATH, show=True, device="cpu", savefig=False)
    # Convert patch to tensor
    tensor_patch_src = convert_patch_to_tensor(patch_src)

    # VAE encoder testing
    encoder = VAEncoder()
    mu, logvar = encoder(tensor_patch_src)

    # Show results
    print(f"Mean shape: {mu.shape}, Log Variance shape: {logvar.shape}")
    fig, ax = plt.subplots(2, 4, figsize=(12, 4))
    for i in range(2):
        if i == 0:
            for j in range(len(mu[0])):
                ax[i, j].imshow(mu[0][j].detach().numpy())
                ax[i, j].axis('off')
                ax[i, j].set_title(f'Mean {j}')
        else:
            for j in range(len(logvar[0])):
                ax[i, j].imshow(logvar[0][j].detach().numpy())
                ax[i, j].axis('off')
                ax[i, j].set_title(f'Log Variance {j}')
    # plt.savefig("assets/vae_output.png")

    cross_attn_vae = ConditionedVAEncoder()
    conditioned_z = cross_attn_vae(tensor_patch_src, style_embedding)

    # Show results
    print(f"Conditioned Z shape: {conditioned_z.shape}")
    _, ax = plt.subplots(1, 4, figsize=(12, 4))
    for j in range(len(conditioned_z[0])):
        ax[j].imshow(conditioned_z[0][j].detach().numpy())
        ax[j].axis('off')
        ax[j].set_title(f'Conditioned Z {j}')
    # plt.savefig("assets/cond_vae_output.png")

    # Blocks testing
    timestep_embedding(torch.tensor([0, 200, 400, 1000]), 4)
    print("Timestep embedding shape:", timestep_embedding(torch.tensor([0, 200, 400, 1000]), 4).shape)

    # Test upsample
    up = Upsample(4)
    tmp = up(conditioned_z)

    # Show results
    print(f"Upsampled conditioned Z shape: {tmp.shape}")
    _, ax = plt.subplots(1, 4, figsize=(12, 4))
    for j in range(len(tmp[0])):
        ax[j].imshow(tmp[0][j].detach().numpy())
        ax[j].axis('off')
        ax[j].set_title(f'Upsampled conditioned Z {j}')
    # plt.savefig("assets/upsampled_cond_vae_output.png")

    down = Downsample(4)
    tmp = down(conditioned_z)

    # Show results
    print(f"Upsampled conditioned Z shape: {tmp.shape}")
    _, ax = plt.subplots(1, 4, figsize=(12, 4))
    for j in range(len(tmp[0])):
        ax[j].imshow(tmp[0][j].detach().numpy())
        ax[j].axis('off')
        ax[j].set_title(f'Upsampled conditioned Z {j}')
    # plt.savefig("assets/downsampled.png")

    # ResBlock testing
    # Create a test tensor (example input)
    batch_size = 1
    in_channels = 4
    out_channels = 8
    dim = 16  # Embedding dimension

    residual_block = ResBlock(
        emb_channels=dim, 
        dropout=0.5, 
        in_channels=in_channels, 
        out_channels=out_channels, 
        use_conv=True, 
        up=True  # Try with up-sampling
    )
    # Define test parameters
    batch_size = 1
    in_channels = 4
    emb_channels = 256
    height, width = 32, 32
    dropout = 0.1

    _, ax = plt.subplots(1, 4, figsize=(12, 4))

    # Create test inputs
    x = conditioned_z  # Input feature map
    emb = style_embedding  # Embedding vector

    # Initialize the block (without up/downsampling)
    res_block = ResBlock(
        emb_channels=emb_channels,
        dropout=dropout,
        in_channels=in_channels,
        use_conv=False,
        out_channels=4,  # Same as in_channels for skip connection identity
        up=False,
        down=False,
        use_scale_shift_norm=True
    )

    # Forward pass
    output = res_block(x, emb)

    # Check output shape
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([1, 4, 32, 32])
    ax[0].imshow(output[0][0].detach().numpy())

    # Additional test with upsampling
    res_block_up = ResBlock(
        emb_channels=emb_channels,
        dropout=dropout,
        in_channels=in_channels,
        use_conv=False,
        out_channels=4,
        up=True,
        use_scale_shift_norm=True
    )
    ax[0].set_title("No conv")

    output_up = res_block_up(x, emb)
    print(f"Output (up) shape: {output_up.shape}")  # Expected: torch.Size([1, 4, 64, 64])
    ax[1].imshow(output_up[0][0].detach().numpy())
    ax[1].set_title("Upsampled")

    # Additional test with downsampling
    res_block_down = ResBlock(
        emb_channels=emb_channels,
        dropout=dropout,
        in_channels=in_channels,
        use_conv=False,
        out_channels=4,
        down=True
    )

    output_down = res_block_down(x, emb)
    print(f"Output (down) shape: {output_down.shape}")  # Expected: torch.Size([1, 4, 16, 16])
    ax[2].imshow(output_down[0][0].detach().numpy())
    ax[2].set_title("Downsampled")

    # Additional test with downsampling
    res_block_down = ResBlock(
        emb_channels=emb_channels,
        dropout=dropout,
        in_channels=in_channels,
        use_conv=False,
        out_channels=8,  # Now correctly set to 8
        up=True,  # Only upsampling
        use_scale_shift_norm=True
    )
    ax[3].set_title("Upsampled 8 out_channels")

    output_down = res_block_down(x, emb)
    print(f"Output (up) shape: {output_down.shape}")  # Expected: torch.Size([1, 4, 16, 16])
    ax[3].imshow(output_down[0][0].detach().numpy())

    # plt.savefig("assets/resblock_output_tmp.png")

    # UNet testing
    model_params = {
        'out_channels': 4,
        'model_channels': 32,
        'num_res_blocks': 2,
        'dropout': 0.1,
        'in_channels': 4,
        'image_size': 32,
        'use_scale_shift_norm': True,
        'resblock_updown': False,  # Disable excessive downsampling
        'num_classes': None,
        'channel_mult': (1, 2, 4),  # Reduce max depth
        # 'device': torch.device('cpu'),
        # 'dtype': torch.float32
    }

    # Create the model
    model = UNetModel(**model_params)

    print("Model initialized successfully!")

    # Test the model
    conditioned_z = torch.randn(1, 4, 32, 32)  # Ensure correct input
    timesteps = torch.tensor([0, 200, 400, 1000])

    # Debug: Print shapes at each layer
    output = model(conditioned_z, timesteps)

    print("Output shape:", output.shape)

    _, ax = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(output[i, j].detach().numpy())
            ax[i, j].axis('off')
            ax[i, j].set_title(f'Output {[i, j]}')

    # plt.savefig("assets/unet_output.png")
    plt.show()