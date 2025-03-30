from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np 
import random 
random.seed(42)

# import os
# os.chdir("..")
from preprocess.utils import read_image, get_tissue_mask, get_image_patches, resize_patch, normalize_patch, convert_patch_to_tensor


class PatchDataset(Dataset):
    def __init__(self, path, tissue_mask_params, patch_extraction_params, domain='A', split='train', test_ratio=0.2, seed=42):
        """
        Dataset for patch extraction with train/test splits for CycleGAN.
        
        Args:
            path: Path to image
            tissue_mask_params: Parameters for tissue mask generation
            patch_extraction_params: Parameters for patch extraction
            domain: 'A' or 'B' to specify the domain
            split: One of 'train' or 'test'
            test_ratio: Ratio of test data (0-1)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.image = read_image(path)
        self.tissue_mask = get_tissue_mask(image=self.image, **tissue_mask_params)
        
        # Extract all patches
        self.patches = get_image_patches(
            image=self.image,
            tissue_mask=self.tissue_mask,
            **patch_extraction_params
        )
        
        # Create indices for train/test split
        indices = np.arange(len(self.patches))
        
        # Split into train and test sets
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
        
        # Select the appropriate indices based on the split
        if split == 'train':
            self.selected_indices = train_indices
        elif split == 'test':
            self.selected_indices = test_indices
        else:
            raise ValueError(f"Split must be one of 'train' or 'test', got {split}")
        
        # Select patches based on indices
        self.patches = [self.patches[i] for i in self.selected_indices]
        
        # Preprocess patches
        self.processed_patches = []
        for patch in self.patches:
            processed = resize_patch(patch)
            processed = normalize_patch(processed)
            self.processed_patches.append(processed)
        
        self.processed_patches = convert_patch_to_tensor(self.processed_patches)
        self.domain = domain
        
    def __len__(self):
        return len(self.processed_patches)

    def __getitem__(self, idx):
        return self.processed_patches[idx]


class CycleGANPatchDataLoader:
    def __init__(self, path_a, path_b, tissue_mask_params, patch_extraction_params, 
                 batch_size=32, test_ratio=0.2, seed=42):
        """
        Creates DataLoaders for trainA, trainB, testA, and testB sets for CycleGAN.
        
        Args:
            path_a: Path to domain A image
            path_b: Path to domain B image
            tissue_mask_params: Parameters for tissue mask generation
            patch_extraction_params: Parameters for patch extraction
            batch_size: Batch size for DataLoaders
            test_ratio: Ratio of test data (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            A dictionary with 'trainA', 'trainB', 'testA', and 'testB' DataLoaders
        """
        # Create datasets for domain A
        self.train_a_dataset = PatchDataset(
            path_a, tissue_mask_params, patch_extraction_params, 
            domain='A', split='train', test_ratio=test_ratio, seed=seed
        )
        self.test_a_dataset = PatchDataset(
            path_a, tissue_mask_params, patch_extraction_params, 
            domain='A', split='test', test_ratio=test_ratio, seed=seed
        )
        
        # Create datasets for domain B
        self.train_b_dataset = PatchDataset(
            path_b, tissue_mask_params, patch_extraction_params, 
            domain='B', split='train', test_ratio=test_ratio, seed=seed
        )
        self.test_b_dataset = PatchDataset(
            path_b, tissue_mask_params, patch_extraction_params, 
            domain='B', split='test', test_ratio=test_ratio, seed=seed
        )
        
        # Create data loaders
        self.loaders = {
            'trainA': DataLoader(self.train_a_dataset, batch_size=batch_size, shuffle=True),
            'trainB': DataLoader(self.train_b_dataset, batch_size=batch_size, shuffle=True),
            'testA': DataLoader(self.test_a_dataset, batch_size=batch_size, shuffle=False),
            'testB': DataLoader(self.test_b_dataset, batch_size=batch_size, shuffle=False)
        }
    
    def get_loaders(self):
        """
        Returns the dictionary of data loaders.
        """
        return self.loaders