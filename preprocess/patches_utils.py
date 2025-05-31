# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

from PIL import Image
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.patches import Rectangle
import subprocess
import os
from collections import defaultdict

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
    def get_image_patches_full(
        image: np.ndarray, 
        patch_size: int = 512,
        is_visualize: bool = False
    ) -> List[np.ndarray]:
        h, w, _ = image.shape

        # Calculate padding only if needed
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)

        if is_visualize:
            fig, ax = plt.subplots()
            ax.imshow(Image.fromarray(image_padded))

        patches = []
        for y in range(0, image_padded.shape[0], patch_size):
            for x in range(0, image_padded.shape[1], patch_size):
                patch = image_padded[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
                
                if is_visualize:
                    rect = Rectangle((x, y), patch_size, patch_size, linewidth=1, edgecolor='green', facecolor='none')
                    ax.add_patch(rect)

        if is_visualize:
            plt.title(f"Total patches: {len(patches)}")
            plt.show()

        return patches

    @staticmethod
    def replace_patches_in_image_full(
        original_image: np.ndarray,
        generated_patches: List[np.ndarray],
        patch_size: int = 512
    ) -> np.ndarray:
        h, w, _ = original_image.shape

        # Compute padding if needed
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        image_padded = np.pad(original_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)

        # Create a copy to modify
        reconstructed_image = image_padded.copy()

        # Replace patches in the same order as full-grid extraction
        patch_idx = 0
        for y in range(0, image_padded.shape[0], patch_size):
            for x in range(0, image_padded.shape[1], patch_size):
                if patch_idx < len(generated_patches):
                    reconstructed_image[y:y + patch_size, x:x + patch_size] = generated_patches[patch_idx]
                    patch_idx += 1
                else:
                    print("Warning: Not enough patches to fill the image. Some areas remain unchanged.")
                    break

        # Remove padding to return to original size
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
    def get_full_info(
            image: np.ndarray, 
            tissue_mask: np.ndarray,
            bbox_info: np.ndarray = None,
            patch_size: int = 512,
            patch_tissue_threshold: float = 0.7,
            min_tissue_threshold: float = 0.1,
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        patch_threshold = int(patch_size * patch_size * patch_tissue_threshold)
        min_patch_threshold = int(patch_size * patch_size * min_tissue_threshold)

        # Image and tissue mask pre-processing
        h, w, _ = image.shape
        pad_b = patch_size - h % patch_size if h % patch_size != 0 else 0
        pad_r = patch_size - w % patch_size if w % patch_size != 0 else 0
        
        image_ = np.pad(image, ((0, pad_b), (0, pad_r), (0, 0)), mode='constant', constant_values=255)
        tissuemask = np.pad(tissue_mask, ((0, pad_b), (0, pad_r)), mode='constant', constant_values=0)

        patches = []
        bbox_patches = []

        def _extract_patches_simple(image_, tissuemask, patch_size, patch_threshold, min_patch_threshold):
            """Extract patches without bbox optimization"""
            patches_data = []
            
            for y in range(0, image_.shape[0], patch_size):
                for x in range(0, image_.shape[1], patch_size):
                    tissuepatch = tissuemask[y:y + patch_size, x:x + patch_size]
                    tissue_pixels = np.sum(tissuepatch)
                    
                    if tissue_pixels > patch_threshold or tissue_pixels > min_patch_threshold:
                        patches_data.append({
                            'coords': (y, x),
                            'bboxes': np.array([])
                        })
            
            return patches_data

        def _extract_patches_with_bbox_optimization(image_, tissuemask, bbox_info, patch_size, 
                                                patch_threshold, min_patch_threshold):
            """Optimized patch extraction with bbox spatial indexing"""
            
            # Create spatial index for bounding boxes
            bbox_grid = defaultdict(list)
            
            for i, bbox in enumerate(bbox_info):
                y0, x0, y1, x1 = bbox[:4]  # Only take first 4 coordinates
                
                # Find which patch grid cells this bbox overlaps with
                start_patch_y = int(y0 // patch_size)
                end_patch_y = int(y1 // patch_size)
                start_patch_x = int(x0 // patch_size)  
                end_patch_x = int(x1 // patch_size)
                
                # Add bbox to all overlapping patches
                for patch_y in range(start_patch_y, end_patch_y + 1):
                    for patch_x in range(start_patch_x, end_patch_x + 1):
                        bbox_grid[(patch_y, patch_x)].append(i)
            
            patches_data = []
            
            # Extract patches and their corresponding bboxes
            for y in range(0, image_.shape[0], patch_size):
                for x in range(0, image_.shape[1], patch_size):
                    tissuepatch = tissuemask[y:y + patch_size, x:x + patch_size]
                    tissue_pixels = np.sum(tissuepatch)
                    
                    if tissue_pixels > patch_threshold or tissue_pixels > min_patch_threshold:
                        patch_y_idx = y // patch_size
                        patch_x_idx = x // patch_size
                        
                        # Get candidate bboxes for this patch
                        candidate_bbox_indices = bbox_grid.get((patch_y_idx, patch_x_idx), [])
                        
                        # Filter bboxes that actually overlap with this patch
                        overlapping_bboxes = []
                        patch_x1, patch_y1 = x + patch_size, y + patch_size
                        
                        for bbox_idx in candidate_bbox_indices:
                            bbox = bbox_info[bbox_idx]
                            y0, x0, y1, x1 = bbox[:4]
                            
                            # Check if bbox overlaps with patch (more lenient condition)
                            if not (x1 <= x or x0 >= patch_x1 or y1 <= y or y0 >= patch_y1):
                                # Adjust bbox coordinates to be relative to patch
                                rel_y0 = max(0, y0 - y)
                                rel_x0 = max(0, x0 - x)  
                                rel_y1 = min(patch_size, y1 - y)
                                rel_x1 = min(patch_size, x1 - x)
                                
                                # Create new bbox with relative coordinates
                                relative_bbox = bbox.copy()
                                relative_bbox[0] = rel_y0
                                relative_bbox[1] = rel_x0
                                relative_bbox[2] = rel_y1
                                relative_bbox[3] = rel_x1
                                
                                overlapping_bboxes.append(relative_bbox)
                        
                        patches_data.append({
                            'coords': (y, x),
                            'bboxes': np.array(overlapping_bboxes) if overlapping_bboxes else np.array([]).reshape(0, bbox_info.shape[1])
                        })
            
            return patches_data
        
        # Pre-compute patch grid if we have bounding boxes
        if bbox_info is not None and len(bbox_info) > 0:
            bbox_patches = _extract_patches_with_bbox_optimization(
                image_, tissuemask, bbox_info, patch_size, 
                patch_threshold, min_patch_threshold
            )
        else:
            bbox_patches = _extract_patches_simple(
                image_, tissuemask, patch_size, 
                patch_threshold, min_patch_threshold
            )
        
        # Extract image patches and corresponding bboxes
        for patch_data in bbox_patches:
            y, x = patch_data['coords']
            patches.append(image_[y:y + patch_size, x:x + patch_size])
        
        # Return bboxes in the same order as patches
        bboxes = [patch_data['bboxes'] for patch_data in bbox_patches]
        
        return patches, bboxes

    # Optional: Helper function to visualize patches with bboxes
    @staticmethod
    def visualize_patch_with_bboxes(patch, bboxes, patch_idx=0):
        """Visualize a patch with its bounding boxes"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches_viz
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(patch)
        
        colors = {
                0 : 'r',    # NEG nuclei
                1 : 'b'     # POS nuclei
            }
        
        for i, bbox in enumerate(bboxes):
            if len(bbox) >= 4:
                y0, x0, y1, x1, label = bbox
                rect = patches_viz.Rectangle((x0, y0), x1-x0, y1-y0, 
                                        linewidth=2, edgecolor=colors[label], 
                                        facecolor='none')
                ax.add_patch(rect)
        
        ax.set_title(f'Patch {patch_idx} with {len(bboxes)} bounding boxes')
        plt.show()