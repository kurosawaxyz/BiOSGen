# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

from PIL import Image
import torch 
from transformers import AutoModelForCausalLM 

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.patches import Rectangle
import subprocess
import os

from .tissue_mask import GaussianTissueMask

class PatchesUtilities:
    def __init__(self):
        pass

    @staticmethod
    def read_image(
        image_fpath: str,
        mode: str = "RGB"
    ) -> np.ndarray:
        with Image.open(image_fpath) as img:
            return np.array(img.convert(mode))

    @staticmethod
    def plot_image_1d(
        image: np.ndarray, 
        save_fpath: str = None, 
        is_show: bool = True
    ) -> None:
        _, ax = plt.subplots()
        ax.imshow(image)
        plt.axis('off')
        
        if save_fpath is not None:
            plt.savefig(save_fpath, bbox_inches='tight', pad_inches=0, dpi=600)

        if is_show:
            plt.show()
    
    @staticmethod
    def plot_image(
        image: np.ndarray, 
        save_fpath: str = None, 
        is_show: bool = True
    ) -> None:
        pil_img = Image.fromarray(image)
        
        if save_fpath is not None:
            pil_img.save(save_fpath)

        if is_show:
            # Notice: actually not working on server
            try:
                subprocess.run(['xdg-open', save_fpath], check=True)
            except Exception as e:
                print(f"Could not open image viewer: {e}")
    
    @staticmethod
    def resize_patch(
        image: np.ndarray, 
        image_size: int = 512
    ) -> np.ndarray:
        img = cv2.resize(image, (image_size, image_size))
        return img

    @staticmethod
    def normalize_patch(
        image: np.ndarray
    ) -> np.ndarray:
        norm_img = cv2.normalize(
            image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        return norm_img


    @staticmethod
    def get_tissue_mask(
        image: np.ndarray,
        kernel_size: int = 20,
        sigma: int = 20,
        downsample: int = 8,
        background_gray_threshold: int = 220,
        **kwargs
    ) -> np.ndarray:
        tissue_detector = GaussianTissueMask(
            kernel_size=kernel_size,
            sigma=sigma,
            downsampling_factor=downsample,
            background_gray_value=background_gray_threshold
        )
        return tissue_detector.process(image)
    
    @staticmethod
    def get_image_patches(
        image: np.ndarray, 
        tissue_mask: np.ndarray,
        patch_size: int = 512,
        patch_tissue_threshold: float = 0.7,
        min_tissue_threshold: float = 0.1,  # New parameter for boundary patches
        is_visualize: bool = False
    ) -> List[np.ndarray]:
        patch_threshold = int(patch_size * patch_size * patch_tissue_threshold)
        min_patch_threshold = int(patch_size * patch_size * min_tissue_threshold)  # Lower threshold for boundary patches
        
        # image and tissue mask pre-processing
        h, w, _ = image.shape
        pad_b = patch_size - h % patch_size
        pad_r = patch_size - w % patch_size
        image_ = np.pad(image, ((0, pad_b), (0, pad_r), (0, 0)), mode='constant', constant_values=255)
        tissue_mask_ = np.pad(tissue_mask, ((0, pad_b), (0, pad_r)), mode='constant', constant_values=0)
        
        if is_visualize:
            _, ax = plt.subplots()
            ax.imshow(Image.fromarray(image_))
        
        # extract patches
        patches = []
        for y in range(0, image_.shape[0], patch_size):
            for x in range(0, image_.shape[1], patch_size):
                tissue_patch_ = tissue_mask_[y:y + patch_size, x:x + patch_size]
                tissue_pixels = np.sum(tissue_patch_)
                
                # Accept patches that meet either the high threshold or the minimum threshold
                if tissue_pixels > patch_threshold or tissue_pixels > min_patch_threshold:
                    patches.append(image_[y:y + patch_size, x:x + patch_size])
                    if is_visualize:
                        # Use different colors to show high vs low tissue content patches
                        color = 'r' if tissue_pixels > patch_threshold else 'orange'
                        rect = Rectangle((x, y), patch_size, patch_size, linewidth=1, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        
        if is_visualize:
            plt.show()
        return patches
    
    @staticmethod
    def replace_patches_in_image(
        original_image: np.ndarray,
        tissue_mask: np.ndarray, 
        generated_patches: List[np.ndarray],
        patch_size: int = 512,
        patch_tissue_threshold: float = 0.7,
        min_tissue_threshold: float = 0.1  # Must match the extraction function
    ) -> np.ndarray:
        patch_threshold = int(patch_size * patch_size * patch_tissue_threshold)
        min_patch_threshold = int(patch_size * patch_size * min_tissue_threshold)
        
        # Same preprocessing as original function
        h, w, _ = original_image.shape
        pad_b = patch_size - h % patch_size
        pad_r = patch_size - w % patch_size
        image_ = np.pad(original_image, ((0, pad_b), (0, pad_r), (0, 0)), mode='constant', constant_values=255)
        tissue_mask_ = np.pad(tissue_mask, ((0, pad_b), (0, pad_r)), mode='constant', constant_values=0)
        
        # Create a copy to modify
        reconstructed_image = image_.copy()
        
        # Replace patches in the same order they were extracted
        patch_idx = 0
        for y in range(0, image_.shape[0], patch_size):
            for x in range(0, image_.shape[1], patch_size):
                tissue_patch_ = tissue_mask_[y:y + patch_size, x:x + patch_size]
                tissue_pixels = np.sum(tissue_patch_)
                
                # Use the same condition as in extraction
                if tissue_pixels > patch_threshold or tissue_pixels > min_patch_threshold:
                    if patch_idx < len(generated_patches):
                        reconstructed_image[y:y + patch_size, x:x + patch_size] = generated_patches[patch_idx]
                        patch_idx += 1
        
        # Remove padding to get back to original dimensions
        return reconstructed_image[:h, :w]
    
    @staticmethod
    def plot_nuclei_labels(
        image: np.ndarray, 
        bbox_info: np.ndarray = None,
        save_fpath: str = None
    ) -> None:
        color_palette = {
            0 : 'r',    # NEG nuclei
            1 : 'b'     # POS nuclei
        }
        _, ax = plt.subplots()
        ax.imshow(Image.fromarray(image))

        for k in range(bbox_info.shape[0]):
            y0, x0, y1, x1, label = bbox_info[k]
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.5, edgecolor=color_palette[label], facecolor='none')
            ax.add_patch(rect)
        
        if save_fpath is not None:
            plt.savefig(save_fpath, dpi=600)

    @staticmethod
    def describe_img(
        image_path: str,
        device: str = "cuda"
    ) -> str:
        img = Image.open(image_path)
        model = load_md(device)
        enc_image = model.encode_image(img)
        desc = model.query(enc_image, "Describe this image.\n")
        print("Description: ", desc)
        return desc



# moondream
def load_md(
    device: str = "cuda"
) -> AutoModelForCausalLM:
    """
    Load the moondream model for image processing.
    Args:
        device (str): Device to load the model on. Default is "cuda".
    Returns:
        AutoModelForCausalLM: Loaded moondream model.

    Caution: device + type must be same as the one used to train the model.
    """
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
    ).to(device)
    return model