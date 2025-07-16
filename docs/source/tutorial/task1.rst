Task 1: Data Preparation
========================

Preliminary
-----------

This task involves preparing the dataset for training the model. It
includes data cleaning, normalization, and splitting the raw histology
images into smaller ready-to-use patches.

The reason why we need to split the images into patches is due to two
main factors: 

1. **Memory Constraints**: Large images can be too big to
   fit into memory, especially when working with high-resolution histology
   images. A histology image contains a lot of information, with over
   5000x5000 pixels. Processing them together, alongside with an
   exponentially larger network, currently there is no GPU that can handle
   this, the fact is that it's not even possible to process to the second
   epoch on a patch of size 1024x1024, which is nearly 5 times smaller than
   the original image.

2. **Model Performance**: Smaller patches allow the model to focus on
   local features, which will indeniably improve the performance of the
   model. Supposing that there exists a GPU that can handle the full
   image, the model will still struggle to learn the local features,
   which are crucial for histology images. The model will be overwhelmed
   by the global features, which are not as important for the task at
   hand.

Repository Structure & Files Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   BiOSGen/
   │── preprocess/            
   │   ├── __init__.py      
   │   ├── dataloader.py            
   │   ├── patches_utils.py    
   │   ├── tissue_mask.py   
   | ..

In this repository, the ``preprocess`` folder contains all the necessary
files for data preparation. The main files are:

- ``dataloader.py``: Contains the ``AntibodiesTree`` class for loading,
  connecting linked files over directories, and create a tree to record
  links to the images, their corresponding masks, and associated
  bbox_informations.
- ``patches_utils.py``: Provides utilities for images and patches
  manipulation.
- ``tissue_mask.py``: Contains helper function to extract patches in the
  mask region applied on the histology images.

These files work together to ensure that the data is properly prepared
for training the model.

Environment Setup
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    chmod +x scripts/setup_env.sh
    ./scripts/setup_env.sh

Data Installation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    chmod +x scripts/data_installer.sh
    ./scripts/data_installer.sh

Data Visualization
------------------

To visualize the data, we need to follow several steps: 

1. **Load the Dataset**: Use the ``AntibodiesTree`` class from ``dataloader.py`` to
   load the dataset and create a tree structure that links images, masks,
   and bounding box information. 
2. **Extract Patches**: Use the ``patches_utils.py`` to extract patches from the images and masks. 
3. **Visualize Patches**: Use a visualization library like Matplotlib to
   display the extracted patches.

**Import libraries:**

.. code-block:: python

    # Basic libraries
    import os
    from PIL import Image
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Personalized modules
    from preprocess.dataloader import AntibodiesTree
    from preprocess.patches_utils import PatchesUtilities

**Hyperparameter:**

.. code-block:: python

    data_dir = "<data_directory>"
    original_stain = "<original_stain>"
    style_stain = "<style_stain>"

Step 1: Load the Dataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    tree_src = AntibodiesTree(
        image_dir = os.path.join(data_dir, original_stain),
        mask_dir = os.path.join(data_dir, "tissue_masks", original_stain),
        npz_dir = os.path.join(data_dir, "bbox_info", f"{original_stain}_{style_stain}", original_stain)
    )
    
    # DST antibodies
    tree_dst = AntibodiesTree(
        image_dir = os.path.join(data_dir, style_stain),
        mask_dir = os.path.join(data_dir, "tissue_masks", style_stain),
        npz_dir = os.path.join(data_dir, "bbox_info", f"{original_stain}_{style_stain}", style_stain)
    )

.. code-block:: python

    print("Nb antibodies: ", tree_src.get_nb_antibodies())
    print("Nb antibodies: ", tree_dst.get_nb_antibodies())

Step 2 + 3: Extract Patches + Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tissue mask associated to the first source and style antibody:**

.. code-block:: python

    tissue_mask_src = PatchesUtilities.get_tissue_mask(image=Image.open(tree_src.antibodies[0]))
    
    tissue_mask_dst = PatchesUtilities.get_tissue_mask(image=Image.open(tree_dst.antibodies[0]))

**Extract patches from the first source and style antibody:**

.. code-block:: python

    patches_src = PatchesUtilities.get_image_patches(
        image=Image.open(tree_src.antibodies[0]),
        tissue_mask=tissue_mask_src,
        is_visualize=True
    )
    
    patches_dst = PatchesUtilities.get_image_patches(
        image=Image.open(tree_dst.antibodies[0]),
        tissue_mask=tissue_mask_dst,
        is_visualize=True
    )
