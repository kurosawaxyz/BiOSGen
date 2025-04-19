<div align="center">
  
  <a href="#"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&duration=3000&pause=1000&color=00FF00&center=true&vCenter=true&width=435&lines=BiOSGen&size=40" alt="Typing SVG" /></a>
</div>

This project presents the work conducted during my first-year master's internship at the ***Paris Brain Institute*** (ICM), in association with the ***DU Sorbonne Data Analytics*** program at ***Université Paris 1 Panthéon-Sorbonne***.

The primary objective was to develop multi-scale image preprocessing and analysis techniques, as well as an integration and fusion method for image analysis. Specifically, this work focused on tumor staining style transfer from H&E to IHC images, aiming to reduce financial costs and save time for more technical analyses.

Two style transfer models were compared: one using a traditional GAN-based approach and another leveraging a more advanced diffusion-based architecture.

Future work will focus on refining the diffusion model to generate more accurate IHC images while incorporating patient-specific health factors that may influence tumor growth, progression, and treatment response.

## Model architecture
<div align="center">

  <img src="assets/diagram.png" alt="architecture" width="750"/>

</div>

## Project structure
```txt
BiOSGen/
│── preprocess/            
│   ├── __init__.py              
│   ├── tissue_mask.py      
│   ├── utils.py  
│── osgen/                   
│   ├── __init__.py             
│   ├── dataloader.py   
│   ├── loss.py
│   ├── nn.py
│   ├── pipeline.py         # Main pipeline for OSGen
│   ├── unet.py
│   ├── vae.py
│   ├── vit.py      
│── configs/               
│   ├── config.yml
│   ├── train_config.yml          
│   ├── test_config.yml        
│── bin/                
│   ├── train.py           
│   ├── eval.py
│   ├── test.py     
│── scripts/                
│   ├── launch.sh     
│── environment.yml
│── README.md               
│── .gitignore  
│── .gitattributes             
```

## Users manual
### Setup virtual environment
#### For miniconda3 users:
```bash
conda env create -f environment.yml
conda activate biosgen
```
#### For miniforge users:
**Warning**:

You will need to remove line 4 and lines 32-34 from `environment.yml` to avoid conflicts while setting up the environment using YAML file. 

As you can't access several `conda` packages in miniforge, you will not be able to use the Moondream model from *vikhyatk/moondream2*. However, there are alternatives such as *HuggingFaceTB/SmolVLM-Instruct*, *microsoft/OmniParser-v2.0*, etc. 

```bash
conda env create -f environment.yml -k
conda activate biosgen
```

#### Important notice
**Warning**:

1. Miniconda environement setup for MacOS has been removed (starting Apr 19 2025), the current version is written for Linux with CUDA. Make sure you're on Linux and have CUDA installed before executing the YAML file. 

2. Please note that issues during conda environment creation may arise due to several operating system incompability or other reasons. As for present, we have encounter OS issues for Mac M1, we suspect it might be due to new Licence for xcode setup on MacOS or Conda 25.01 Licence *(but Conda Licence should not be this severe)*. The first issue is due to `cmake` and `protobuf` missing, make it unable to create wheels for `pyproject.toml` for `onnx`. In this case, you need to:

```shell
# Install Homebrew if it haven't been done (remember to follow given instructions)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Check if brew is well established
brew --version

# Install cmake and protobuf via brew
brew install cmake protobuf

# Rerun pip setup
pip install -r requirements.txt
```



### Data installation
>**Note**: Data installation instruction for the EMPaCT dataset provided by [AI4SCR](https://github.com/AI4SCR/VirtualMultiplexer)
#### Downloading the EMPaCT dataset 

European Multicenter Prostate Cancer Clinical and Translational Research Group (EMPaCT) [[1](https://www.sciencedirect.com/science/article/pii/S0022534712029710), [2](https://www.sciencedirect.com/science/article/pii/S2405456917300020), [3](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2020.00246/full)] contains prostate cancer tissue microarrays (TMAs) from 210 patients with 4 cores per patient for several clinically relevant markers.

All images from Hematoxylin & Eosin (H&E) and Immunohistochemistry (IHC) stainings are uploaded to Zenodo under a Creative Commons Attribution 4.0 International License and can be dowloaded from this [link](https://zenodo.org/records/10066853).

#### Downloading Masks, Bounding Boxes and Data Splits

AI4SCR uploaded the all relevant information to [Figshare]( https://figshare.com/projects/VirtualMultiplexer/230498) and 
provide a notebook to demonstrate how to process and plot the data.
