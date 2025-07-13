Welcome to BiOSGen's documentation!
===================================

What is BiOSGen? 
===============

BiOSGen is a Pipeline for Clinical Neural Style Transfer for Tumor Staining with Stable Diffusion Models.

The primary objective was to develop multi-scale image preprocessing and analysis techniques, as well as an integration and fusion method for image analysis. Specifically, this work focused on tumor staining style transfer from H&E to IHC images, aiming to reduce financial costs and save time for more technical analyses.

And why do we build it? 
------------------------
Tumor staining is a time- and cost-intensive process due to several factors. Traditional staining protocols, such as *Hematoxylin and Eosin* (H&E) and *Immunohistochemistry* (IHC), require precise chemical preparation, specialized reagents, and strict protocols to ensure consistency. Staining quality can vary due to differences in reagent concentrations, incubation times, tissue thickness, and manual handling, leading to inconsistencies in diagnostics.

Furthermore, the high cost of antibodies and reagents, combined with the labor-intensive nature of manual staining, makes large-scale analysis expensive and inefficient.

When generative models are applied for automated staining conversion, these factors introduce additional challenges. A model must learn complex mappings between H&E and IHC stains while accounting for tissue morphology variations, staining intensity, and potential artifacts.

Additionally, because histopathological images contain critical diagnostic information, the model must preserve both visual fidelity and biological relevance, avoiding hallucinations or distortions that could lead to misdiagnosis. Patient-specific health factors, such as tumor type, stage, or underlying conditions, can also influence staining characteristics, adding another layer of complexity (ref. [roy_cancer_2018]_).

These challenges make the development of a reliable and effective computational staining approach particularly difficult.

.. .. image:: ../assets/diagram.png
..    :alt: architecture
..    :align: center
..    :width: 750

.. toctree::
   :maxdepth: 2
   :caption: Summary:

   introduction
   quickstart
   installation
   tutorial/index