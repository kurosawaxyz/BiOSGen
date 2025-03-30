import torch
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import os

# Primarily use osgen DataLoader
from osgen.dataloader import PatchDataLoader

import argparse

if __name__ == "__main__":

    # Load argparger
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Argparse for training process.")

    # Add arguments
    parser.add_argument("--config_path", type=str, help="Configuration path", required=True)
    parser.add_argument("--style_path", type=str, help="Style tumor path", required=True)
    parser.add_argument("--original_path", type=str, help="Original tumor path", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)


    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    device = cfg.device
    # learning_rate = cfg.learning_rate

    tissue_mask_params = cfg.Image.tissue_mask_params
    patch_extraction_params = cfg.Image.patch_extraction_params
    dataset_path = args.dataset_path

    # Remove existing data
    # os.remove(dataset_path)

    IMAGE_PATH = args.original_path
    STYLE_PATH = args.style_path
    
    data_loader = PatchDataLoader(
        path_src=IMAGE_PATH,
        path_dst=STYLE_PATH,
        tissue_mask_params=tissue_mask_params,
        patch_extraction_params=patch_extraction_params,
        batch_size=3,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Print information about each split
    print(f"Train dataset size: {len(data_loader.train_dataset)}")
    print(f"Validation dataset size: {len(data_loader.val_dataset)}")
    print(f"Test dataset size: {len(data_loader.test_dataset)}")


    # Create dir
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    dir_train = os.path.join(dataset_path, "train")
    dir_val = os.path.join(dataset_path, "val")
    dir_test = os.path.join(dataset_path, "test")

    if not os.path.exists(dir_train):
        os.mkdir(dir_train)
    if not os.path.exists(dir_val):
        os.mkdir(dir_val)
    if not os.path.exists(dir_test):
        os.mkdir(dir_test)

    # Save data train
    if not os.path.exists(f"{dir_train}/original"):
        os.mkdir(f"{dir_train}/original")
    if not os.path.exists(f"{dir_train}/style"):
        os.mkdir(f"{dir_train}/style")

    for i, (src, dst) in enumerate(data_loader.train_dataset):
        # Convert tensors to PIL images for visualization
        src_image = Image.fromarray(src.permute(1, 2, 0).detach().numpy().astype(np.uint8)).save(f"{dir_train}/original/original_train_{i}.png")
        dst_image = Image.fromarray(dst[0].permute(1, 2, 0).detach().numpy().astype(np.uint8)).save(f"{dir_train}/style/style_train_{i}.png")

    # Save data val
    if not os.path.exists(f"{dir_val}/original"):
        os.mkdir(f"{dir_val}/original")
    if not os.path.exists(f"{dir_val}/style"):
        os.mkdir(f"{dir_val}/style")
    for i, (src, dst) in enumerate(data_loader.val_dataset):
        # Convert tensors to PIL images for visualization
        src_image = Image.fromarray(src.permute(1, 2, 0).detach().numpy().astype(np.uint8)).save(f"{dir_val}/original/original_val_{i}.png")
        dst_image = Image.fromarray(dst[0].permute(1, 2, 0).detach().numpy().astype(np.uint8)).save(f"{dir_val}/style/style_val_{i}.png")

    # Save data test
    if not os.path.exists(f"{dir_test}/original"):
        os.mkdir(f"{dir_test}/original")
    if not os.path.exists(f"{dir_test}/style"):
        os.mkdir(f"{dir_test}/style")
    for i, (src, dst) in enumerate(data_loader.test_dataset):
        # Convert tensors to PIL images for visualization
        src_image = Image.fromarray(src.permute(1, 2, 0).detach().numpy().astype(np.uint8)).save(f"{dir_test}/original/original_test_{i}.png")
        dst_image = Image.fromarray(dst[0].permute(1, 2, 0).detach().numpy().astype(np.uint8)).save(f"{dir_test}/style/style_test_{i}.png")
        

print("Finished saving data.")