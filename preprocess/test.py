from preprocess import * 
import numpy as np
import os

tree = AntibodiesTree(
    image_dir="/root/BiOSGen/data/HE",
    mask_dir="/root/BiOSGen/data/tissue_masks/HE",
    npz_dir="/root/BiOSGen/data/bbox_info/HE_NKX3/HE"
)
print("Nb antibodies",tree.get_nb_antibodies())

# Test utility functions
util = Utilities()
img = util.read_image(
    tree.antibodies[0]
)

print("plot image")
util.plot_image(img, save_fpath="archive/test.png", is_show=False)
print("plot image done")




# Test higher functions
# tissue mask params
tissue_mask_params = {
    'kernel_size': 20,
    'sigma': 20,
    'downsample': 8,
    'background_gray_threshold': 220
}

# patch extraction params
patch_extraction_params = {
    'patch_size': 512,
    'patch_tissue_threshold': 0.7,
    'is_visualize': True
}

tissue_mask_src = util.get_tissue_mask(image=img, **tissue_mask_params)
util.plot_image_1d(tissue_mask_src, save_fpath="archive/tissue_mask.png", is_show=False)


# extract src patches
patches_src = util.get_image_patches(
    image=img,
    tissue_mask=tissue_mask_src,
    **patch_extraction_params
)



# bbox
# read and plot bbox info for src (download the pre-extracted bbox-info from zenodo)
data = np.load(tree.npz[0])
bbox_info = data['bbox']        # (y0, x0, y1, x1, label)
# overlay nuclei labels on image
util.plot_nuclei_labels(image=img, bbox_info=bbox_info)