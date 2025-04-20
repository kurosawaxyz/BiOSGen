import os
import pandas as pd 
from typing import List


class AntibodiesTree:
    def __init__(
            self,
            image_dir: str,
            mask_dir: str,
            npz_dir: str
    ):
        """
        Initialize the AntibodiesTree class.

        Args:
            image_dir (str): Directory containing images.
            mask_dir (str): Directory containing masks.
            npz_dir (str): Directory containing npz files.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.npz_dir = npz_dir
        self._construct_tree()

    def _construct_tree(self) -> None:
        """
        Construct the tree structure for the images, masks, and npz files.

        Returns:
            None
        """
        associations = split_data(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            npz_dir=self.npz_dir,
        )
        # Create a tree structure
        self.antibodies = associations.columns
        self.mask = associations.iloc[0]
        self.npz = associations.iloc[1]

    def get_nb_antibodies(self) -> int:
        """
        Get the number of antibodies.

        Returns:
            int: Number of antibodies.
        """
        return len(self.antibodies)

    def get_antibodies(self) -> List[str]:
        """
        Get the list of antibodies.

        Returns:
            list: List of antibodies.
        """
        return self.antibodies
    
    def get_mask(self) -> List[str]:
        """
        Get the list of masks.

        Returns:
            list: List of masks.
        """
        return self.mask
    def get_npz(self) -> List[str]:
        """
        Get the list of npz files.
        Returns:
            list: List of npz files.
        """
        return self.npz


def split_data(
    image_dir: str,
    mask_dir: str,
    npz_dir: str,
    output_dir: str = None,
) -> None:
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
    if output_dir is not None: 
        df = pd.DataFrame(associations)
        df.to_csv(output_dir, index=False)
        print("Data associations saved to data_associations.csv")

    return pd.DataFrame(associations)
