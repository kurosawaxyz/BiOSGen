Quick Start
===========

To get started quickly:

1. Install the package with `pip install biosgen`
2. Run `biosgen --help` to see available commands.

---------------------------------------------------

Installation
============

Requirements
-------------
- Python 3.8 or higher
- CUDA-enabled GPU (A100 or higher, required for training and faster processing, optional for inference)
- Virtual conda environment (recommended)
- pip

Install with PyPI
-------------------

.. code-block:: bash

    pip install biosgen

Install full code for further development
---------------------------------------------------------

For training or code reproduction, you can clone the repository:

.. code-block:: bash

   git clone https://github.com/kurosawaxyz/BiOSGen.git
   cd biosgen
   pip install -e .

It is recommended to use a virtual environment to avoid conflicts with other packages. You can create a virtual environment using conda:

.. code-block:: bash

   conda create -n biosgen python=3.10 -y
   conda activate biosgen

.. warning::

    **Note**: `-y` flag is used to automatically confirm the installation of packages without prompting for user input. This is useful when you want to install multiple packages in a single command without having to manually confirm each one.

To train with appropriate datasets, it is recommended to install the EmPACT TMA data, published by ETH Zürich on Zenodo. Instructions for downloading and preparing the dataset can be found below:

Data Installation
------------------------

Install Train-Test Data
^^^^^^^^^^^^^^^^^^^^^^^^
To install the training and testing data, run:

.. code-block:: bash

   chmod +x scripts/data_installer.sh
   ./scripts/data_installer.sh

.. note::

   Data installation instructions for the EMPaCT dataset are provided by `AI4SCR <https://github.com/AI4SCR/VirtualMultiplexer>`_.

Downloading the EMPaCT Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **European Multicenter Prostate Cancer Clinical and Translational Research Group (EMPaCT)** dataset [#empact1]_ [#empact2]_ [#empact3]_ contains prostate cancer tissue microarrays (TMAs) from 210 patients, with 4 cores per patient, across several clinically relevant markers.

All H&E and IHC-stained images are available on **Zenodo** under a *Creative Commons Attribution 4.0 International License*.  
You can download them from: https://zenodo.org/records/10066853

Downloading Masks, Bounding Boxes and Data Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AI4SCR has uploaded the relevant masks, bounding boxes, and train-test splits to `Figshare <https://figshare.com/projects/VirtualMultiplexer/230498>`_.  
They also provide a Jupyter notebook to demonstrate how to process and visualize the data.

---------------------------------------------------

.. rubric:: References

.. [#empact1] https://www.sciencedirect.com/science/article/pii/S0022534712029710  
.. [#empact2] https://www.sciencedirect.com/science/article/pii/S2405456917300020  
.. [#empact3] https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2020.00246/full
