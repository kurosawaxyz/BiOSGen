Overview
===================================

The architecture of BiOSGen can be visualized below. It consists of several components that work together to perform consecutive end2end tasks, from data preprocessing to style extraction, tumor generation, and post-processing.

.. image:: ../assets/diagram.png
   :alt: architecture
   :align: center
   :width: 750

Our model consists of four main components:

1. **Data Preprocessing Module**: This component handles the preprocessing of input H&E images, including normalization, resizing, and augmentation to prepare them for style transfer.

2. **Style Extractor**: This component focuses on extracting style embeddings from the input style tumor images.

3. **Style Transfer Backbone**: This component is responsible for performing the actual style transfer from the input H&E images to the target IHC images using the extracted style embeddings, combined with the latent vector of the input images, output of the Variational Autoencoder (VAE).

4. **Post-Processing Module**: This component handles the post-processing of the generated IHC images, where the output of the generative backbone will be adjusted following clinical notices. This characteristics will be released in future updates.

Together, these components form a comprehensive pipeline for automated tumor staining, addressing the challenges outlined earlier.