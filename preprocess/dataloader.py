import os
import pandas as pd 

def split_data(
    image_dir: str,
    mask_dir: str,
    npz_dir: str,
    output_dir: str,
):
    """
    Summarize + associate the data in the image and mask directories

    Args:
        image_dir (str): Directory containing images.
        mask_dir (str): Directory containing masks.
        npz_dir (str): Directory containing npz files.

    Returns:
        None
    """
    # Get file lists
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

    # Create a dictionary to store the associations
    associations = {}

    for image_file in image_files:
        # Extract the common base (e.g., A6_TMA_15_02_IVB_HE)
        base_name = image_file.rsplit('.', 1)[0]  # Remove .png

        # Match npz and mask files that start with this base_name followed by "_"
        matching_npz = next((npz for npz in npz_files if npz.startswith(base_name + '_')), None)
        matching_mask = next((mask for mask in mask_files if mask.startswith(base_name + '_')), None)

        if matching_npz is not None and matching_mask is not None:
            associations[image_file] = {
                'npz': matching_npz,
                'mask': matching_mask
            }
        else: continue

    # Convert to DataFrame
    df = pd.DataFrame(associations)
    df.to_csv(output_dir, index=False)
    print("Data associations saved to data_associations.csv")
