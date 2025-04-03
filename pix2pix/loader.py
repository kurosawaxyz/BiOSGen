import torch
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import os

# Primarily use osgen DataLoader
from pix2pix.dataloader import Pix2PixPatchDataLoader

import argparse

# python -m pix2pix.loader --config_path configs/config.yml --style_path demo/img/A6_TMA_15_02_IVB_NKX.png --original_path demo/img/A4_TMA_15_02_IVB_HE.png --dataset_path pix2pix/data/

if __name__ == "__main__":
    # Load argparser
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Argparse for Pix2Pix dataset preparation.")

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
   
    # Create the Pix2Pix data loader with our new implementation
    data_loader = Pix2PixPatchDataLoader(
        path_a=DOMAIN_A_PATH,
        path_b=DOMAIN_B_PATH,
        tissue_mask_params=tissue_mask_params,
        patch_extraction_params=patch_extraction_params,
        batch_size=3,
        val_ratio=0.15,
        test_ratio=0.15
    )
   
    loaders = data_loader.get_loaders()

    # Print information about each dataset
    print(f"Train dataset size: {len(data_loader.train_dataset)}")
    print(f"Validation dataset size: {len(data_loader.val_dataset)}")
    print(f"Test dataset size: {len(data_loader.test_dataset)}")

    # Create dir
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    # Create only train, val, test directories
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

    # Save train data - only domain A (original) images
    for i, (img_a, _) in enumerate(data_loader.train_dataset):
        img_a_pil = Image.fromarray(img_a.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        img_a_pil.save(f"{dataset_path}/train/img_{i}.png")

    # Save validation data - only domain B (styled) images
    for i, (_, img_b) in enumerate(data_loader.val_dataset):
        img_b_pil = Image.fromarray(img_b.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        img_b_pil.save(f"{dataset_path}/val/img_{i}.png")

    # Save test data - only domain B (styled) images
    for i, (_, img_b) in enumerate(data_loader.test_dataset):
        img_b_pil = Image.fromarray(img_b.permute(1, 2, 0).detach().numpy().astype(np.uint8))
        img_b_pil.save(f"{dataset_path}/test/img_{i}.png")

    print(f"Dataset successfully created at {dataset_path}")
    print("Finished saving data.")