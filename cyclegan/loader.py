import torch
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import os

# Primarily use osgen DataLoader
from cyclegan.dataloader import CycleGANPatchDataLoader

import argparse

if __name__ == "__main__":
    # Load argparser
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Argparse for CycleGAN dataset preparation.")

    # Add arguments
    parser.add_argument("--config_path", type=str, help="Configuration path", required=True)
    parser.add_argument("--original_path", type=str, help="Domain A image path", required=True)
    parser.add_argument("--style_path", type=str, help="Domain B image path", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    device = cfg.device

    tissue_mask_params = cfg.Image.tissue_mask_params
    patch_extraction_params = cfg.Image.patch_extraction_params
    dataset_path = args.dataset_path

    DOMAIN_A_PATH = args.original_path
    DOMAIN_B_PATH = args.style_path
    
    # Create the CycleGAN data loader with our new implementation
    data_loader = CycleGANPatchDataLoader(
        path_a=DOMAIN_A_PATH,
        path_b=DOMAIN_B_PATH,
        tissue_mask_params=tissue_mask_params,
        patch_extraction_params=patch_extraction_params,
        batch_size=3,
        test_ratio=0.2
    )
    
    loaders = data_loader.get_loaders()

    # Print information about each dataset
    print(f"TrainA dataset size: {len(data_loader.train_a_dataset)}")
    print(f"TrainB dataset size: {len(data_loader.train_b_dataset)}")
    print(f"TestA dataset size: {len(data_loader.test_a_dataset)}")
    print(f"TestB dataset size: {len(data_loader.test_b_dataset)}")

    # Create dir
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    # Create directories for the CycleGAN structure
    dirs = ['trainA', 'trainB', 'testA', 'testB']
    for dir_name in dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # Save trainA data
    for i, src in enumerate(data_loader.train_a_dataset):
        src_image = Image.fromarray(src.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        src_image.save(f"{dataset_path}/trainA/domainA_train_{i}.png")

    # Save trainB data
    for i, dst in enumerate(data_loader.train_b_dataset):
        dst_image = Image.fromarray(dst.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        dst_image.save(f"{dataset_path}/trainB/domainB_train_{i}.png")

    # Save testA data
    for i, src in enumerate(data_loader.test_a_dataset):
        src_image = Image.fromarray(src.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        src_image.save(f"{dataset_path}/testA/domainA_test_{i}.png")

    # Save testB data
    for i, dst in enumerate(data_loader.test_b_dataset):
        dst_image = Image.fromarray(dst.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        dst_image.save(f"{dataset_path}/testB/domainB_test_{i}.png")

    print(f"Dataset successfully created at {dataset_path}")
        

print("Finished saving data.")