import torch 
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM 

import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# moondream
def load_md(
    device: str = "cuda"
):
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
    ).to(device)
    return model

def describe_img(
    image_path: str,
    device: str = "cuda"
):
    img = Image.open(image_path)
    model = load_md(device)
    enc_image = model.encode_image(img)
    print(model.query(enc_image, "Describe this image.\n"))

def read_image(
    image_fpath: str
) -> np.ndarray:
    return np.array(Image.open(image_fpath))

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

def get_image_patches(
    image: np.ndarray, 
    tissue_mask: np.ndarray,
    patch_size: int = 512,
    patch_tissue_threshold: float = 0.7,
    is_visualize: bool = True
) -> List[np.ndarray]:
    patch_threshold = int(patch_size * patch_size * patch_tissue_threshold)
    
    # image and tissue mask pre-processing
    h, w, c = image.shape
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
                patches.append(
                    image_[y:y + patch_size, x:x + patch_size, :]
                )
    
                if is_visualize:
                    rect = matplotlib.patches.Rectangle((x, y), patch_size, patch_size,
                                             linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
    
    if is_visualize:
        plt.show()
        
    return patches

def resize_patch(image):
    img = cv2.resize(image, (32, 32))
    return img

def normalize_patch(image):
    norm_img = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    return norm_img

def convert_patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).float()
    return patch