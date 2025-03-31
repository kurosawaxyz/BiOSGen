from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np 
import random 
random.seed(42)

# import os
# os.chdir("..")
from preprocess.utils import read_image, get_tissue_mask, get_image_patches, resize_patch, normalize_patch, convert_patch_to_tensor


class PairedPatchDataset(Dataset):
    def __init__(self, path_a, path_b, tissue_mask_params, patch_extraction_params, split='train', val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Dataset for paired patch extraction with train/val/test splits for Pix2Pix.
       
        Args:
            path_a: Path to domain A image
            path_b: Path to domain B image
            tissue_mask_params: Parameters for tissue mask generation
            patch_extraction_params: Parameters for patch extraction
            split: One of 'train', 'val', or 'test'
            val_ratio: Ratio of validation data (0-1)
            test_ratio: Ratio of test data (0-1)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
       
        self.image_a = read_image(path_a)
        self.image_b = read_image(path_b)
        self.tissue_mask_a = get_tissue_mask(image=self.image_a, **tissue_mask_params)
        self.tissue_mask_b = get_tissue_mask(image=self.image_b, **tissue_mask_params)
       
        # Extract all patches
        self.patches_a = get_image_patches(
            image=self.image_a,
            tissue_mask=self.tissue_mask_a,
            **patch_extraction_params
        )
        self.patches_b = get_image_patches(
            image=self.image_b,
            tissue_mask=self.tissue_mask_b,
            **patch_extraction_params
        )
       
        # If patches from the two domains have different lengths, use the smaller number
        min_patches = min(len(self.patches_a), len(self.patches_b))
        self.patches_a = self.patches_a[:min_patches]
        self.patches_b = self.patches_b[:min_patches]
       
        # Create indices for train/val/test split
        indices = np.arange(min_patches)
       
        # First split: separate test set
        remaining_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
       
        # Second split: separate train and validation sets
        val_ratio_adjusted = val_ratio / (1 - test_ratio)  # Adjust val_ratio to account for test set removal
        train_indices, val_indices = train_test_split(
            remaining_indices, test_size=val_ratio_adjusted, random_state=seed
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
        self.patches_a = [self.patches_a[i] for i in self.selected_indices]
        self.patches_b = [self.patches_b[i] for i in self.selected_indices]
       
        # Preprocess patches
        self.processed_patches_a = []
        self.processed_patches_b = []
       
        for patch_a, patch_b in zip(self.patches_a, self.patches_b):
            processed_a = resize_patch(patch_a)
            processed_a = normalize_patch(processed_a)
            self.processed_patches_a.append(processed_a)
           
            processed_b = resize_patch(patch_b)
            processed_b = normalize_patch(processed_b)
            self.processed_patches_b.append(processed_b)
       
        self.processed_patches_a = convert_patch_to_tensor(self.processed_patches_a)
        self.processed_patches_b = convert_patch_to_tensor(self.processed_patches_b)
       
    def __len__(self):
        return len(self.processed_patches_a)

    def __getitem__(self, idx):
        return self.processed_patches_a[idx], self.processed_patches_b[idx]


class Pix2PixPatchDataLoader:
    def __init__(self, path_a, path_b, tissue_mask_params, patch_extraction_params,
                 batch_size=32, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Creates DataLoaders for train, validation, and test sets for Pix2Pix.
       
        Args:
            path_a: Path to domain A image
            path_b: Path to domain B image
            tissue_mask_params: Parameters for tissue mask generation
            patch_extraction_params: Parameters for patch extraction
            batch_size: Batch size for DataLoaders
            val_ratio: Ratio of validation data (0-1)
            test_ratio: Ratio of test data (0-1)
            seed: Random seed for reproducibility
           
        Returns:
            Dictionary with 'train', 'val', and 'test' DataLoaders
        """
        self.train_dataset = PairedPatchDataset(
            path_a, path_b, tissue_mask_params, patch_extraction_params,
            split='train', val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
       
        self.val_dataset = PairedPatchDataset(
            path_a, path_b, tissue_mask_params, patch_extraction_params,
            split='val', val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
       
        self.test_dataset = PairedPatchDataset(
            path_a, path_b, tissue_mask_params, patch_extraction_params,
            split='test', val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
       
        # Create data loaders
        self.loaders = {
            'train': DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        }
   
    def get_loaders(self):
        """
        Returns the dictionary of data loaders.
        """
        return self.loaders