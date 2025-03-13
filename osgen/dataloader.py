from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np 
import random 
random.seed(42)

# import os
# os.chdir("..")
from preprocess.utils import read_image, get_tissue_mask, get_image_patches, resize_patch, normalize_patch, convert_patch_to_tensor


class PatchDataset(Dataset):
    def __init__(self, path_src, path_dst, tissue_mask_params, patch_extraction_params, split='train', val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Dataset for patch extraction with train/val/test splits.
        
        Args:
            path_src: Path to source image
            path_dst: Path to destination image
            tissue_mask_params: Parameters for tissue mask generation
            patch_extraction_params: Parameters for patch extraction
            split: One of 'train', 'val', or 'test'
            val_ratio: Ratio of validation data (0-1)
            test_ratio: Ratio of test data (0-1)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.image_src = read_image(path_src)
        self.image_dst = read_image(path_dst)
        self.tissue_mask_src = get_tissue_mask(image=self.image_src, **tissue_mask_params)
        self.tissue_mask_dst = get_tissue_mask(image=self.image_dst, **tissue_mask_params)
        
        # Extract all patches
        self.patches_src = get_image_patches(
            image=self.image_src,
            tissue_mask=self.tissue_mask_src,
            **patch_extraction_params
        )
        self.patches_dst = get_image_patches(
            image=self.image_dst,
            tissue_mask=self.tissue_mask_dst,
            **patch_extraction_params
        )
        
        # Create indices for train/val/test split
        indices = np.arange(len(self.patches_src))
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
        
        # Second split: separate train and validation sets
        val_ratio_adjusted = val_ratio / (1 - test_ratio)  # Adjust val_ratio to account for test set removal
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio_adjusted, random_state=seed
        )
        
        # Select the appropriate indices based on the split
        if split == 'train':
            self.selected_indices = train_indices
        elif split == 'val':
            self.selected_indices = val_indices
        elif split == 'test':
            self.selected_indices = test_indices
        else:
            raise ValueError(f"Split must be one of 'train', 'val', or 'test', got {split}")
        
        # Select patches based on indices
        self.patches_src = [self.patches_src[i] for i in self.selected_indices]
        
        # Preprocess patches
        self.processed_patches = []
        for patch in self.patches_src:
            processed = resize_patch(patch)
            processed = normalize_patch(processed)
            self.processed_patches.append(processed)
        
        self.processed_patches = convert_patch_to_tensor(self.processed_patches)
        
    def __len__(self):
        return len(self.processed_patches)

    def __getitem__(self, idx):
        # For the destination patch, randomly select from all available destination patches
        random_dst_idx = np.random.randint(len(self.patches_dst))
        dst_patch = self.patches_dst[random_dst_idx]
        
        # Process the destination patch
        dst_patch = resize_patch(dst_patch)
        dst_patch = normalize_patch(dst_patch)
        dst_patch = convert_patch_to_tensor([dst_patch])
        
        return self.processed_patches[idx], dst_patch


class PatchDataLoader:
    def __init__(self, path_src, path_dst, tissue_mask_params, patch_extraction_params, 
                 batch_size=32, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Creates DataLoaders for train, validation, and test sets.
        
        Returns:
            A dictionary with 'train', 'val', and 'test' DataLoaders
        """
        self.train_dataset = PatchDataset(
            path_src, path_dst, tissue_mask_params, patch_extraction_params, 
            split='train', val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        self.val_dataset = PatchDataset(
            path_src, path_dst, tissue_mask_params, patch_extraction_params, 
            split='val', val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        self.test_dataset = PatchDataset(
            path_src, path_dst, tissue_mask_params, patch_extraction_params, 
            split='test', val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )