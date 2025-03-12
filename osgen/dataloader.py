from torch.utils.data import Dataset, DataLoader

import numpy as np 
import random 
random.seed(42)

from .preprocess.utils import read_image, get_tissue_mask, get_image_patches, resize_patch, normalize_patch, convert_patch_to_tensor


class PatchDataset(Dataset):
    def __init__(self, path_src, path_dst, tissue_mask_params, patch_extraction_params, batch_size):
        self.image_src = read_image(path_src)
        self.image_dst = read_image(path_dst)
        self.tissue_mask_src = get_tissue_mask(image=self.image_src, **tissue_mask_params)
        self.tissue_mask_dst = get_tissue_mask(image=self.image_dst, **tissue_mask_params)
        self.patch_extraction_params = patch_extraction_params
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
        self.batch_size = batch_size
        self.batch = self.patches_src[:batch_size]

        # Preprocess batch
        for i in range(len(self.batch)):
            self.batch[i] = resize_patch(self.batch[i])
            self.batch[i] = normalize_patch(self.batch[i])
        
        self.batch = convert_patch_to_tensor(self.batch)

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        return self.batch[idx], self.patches_dst[np.random.randint(len(self.patches_dst))]        
    
class PatchDataLoader(DataLoader):
    def __init__(self, path_src, path_dst, tissue_mask_params, patch_extraction_params, batch_size):
        dataset = PatchDataset(path_src, path_dst, tissue_mask_params, patch_extraction_params, batch_size)
        super().__init__(dataset, batch_size=batch_size)