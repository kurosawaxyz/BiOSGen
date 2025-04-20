from PIL import Image
import torch 
from transformers import AutoModelForCausalLM 

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.patches as patches

from .tissue_mask import GaussianTissueMask

class Utilities:
    def __init__(self):
        pass

    @staticmethod
    def read_image(
        image_fpath: str
    ) -> np.ndarray:
        return np.array(Image.open(image_fpath))
    
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
        is_visualize: bool = True
    ) -> List[np.ndarray]:
        patch_threshold = int(patch_size * patch_size * patch_tissue_threshold)
        
        # image and tissue mask pre-processing
        h, w, _ = image.shape
        pad_b = patch_size - h % patch_size
        pad_r = patch_size - w % patch_size
        image_ = np.pad(image, ((0, pad_b), (0, pad_r), (0, 0)), mode='constant', constant_values=255)
        tissue_mask_ = np.pad(tissue_mask, ((0, pad_b), (0, pad_r)), mode='constant', constant_values=0)
        
        if is_visualize:
            fig, ax = plt.subplots()
            ax.imshow(Image.fromarray(image_))
        
        # extract patches
        patches = []
        for y in range(0, image_.shape[0], patch_size):
            for x in range(0, image_.shape[1], patch_size):
                tissue_patch_ = tissue_mask_[y:y + patch_size, x:x + patch_size]
                if np.sum(tissue_patch_) > patch_threshold:
                    patches.append(image_[y:y + patch_size, x:x + patch_size])
                    if is_visualize:
                        rect = patches.Rectangle((x, y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
        if is_visualize:
            plt.show()
        return patches
    
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
        fig, ax = plt.subplots()
        ax.imshow(Image.fromarray(image))

        for k in range(bbox_info.shape[0]):
            y0, x0, y1, x1, label = bbox_info[k]
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.5, edgecolor=color_palette[label], facecolor='none')
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