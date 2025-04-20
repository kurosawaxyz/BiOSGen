from preprocess import * 

tree = AntibodiesTree(
    image_dir="/root/BiOSGen/data/HE",
    mask_dir="/root/BiOSGen/data/tissue_masks/HE",
    npz_dir="/root/BiOSGen/data/bbox_info/HE_NKX3/HE"
)
# print(tree.get_nb_antibodies())

# Test utility functions
util = Utilities()
