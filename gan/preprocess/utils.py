from typing import List
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from .tissue_mask import GaussianTissueMask


def create_csv(
    data_type_src="HE", 
    data_type_dst="NKX3", 
    data_dir="/Users/vuhoangthuyduong/Documents/icm/tumor-aug-test/data", 
    **kwargs
):
    df = dict({data_type_src: [], data_type_dst: []})

    #print(f"preprocess::create_csv for {data_type_src} -> {data_type_dst} data")

    try:
        for root, dirs, files in os.walk(data_dir):
            for d in dirs:
                subdirectory = os.path.join(root, d)
                list_files = os.listdir(subdirectory)
                for f in range(500):
                    if f != ".DS_Store":
                        filename = os.path.join(subdirectory, list_files[f])
                        if d == 'HE' : 
                            df[data_type_src].append(filename)
                        else: 
                            df[data_type_dst].append(filename)
    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(df)
    return df


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
    coords = []
    for y in range(0, image_.shape[0], patch_size):
        for x in range(0, image_.shape[1], patch_size):
            tissue_patch_ = tissue_mask_[y:y + patch_size, x:x + patch_size]
            if np.sum(tissue_patch_) > patch_threshold:
                patches.append(
                    image_[y:y + patch_size, x:x + patch_size, :]
                )
                coords.append((y, x))
    
                if is_visualize:
                    rect = matplotlib.patches.Rectangle((x, y), patch_size, patch_size,
                                             linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
    
    if is_visualize:
        plt.show()
        
    return patches, coords