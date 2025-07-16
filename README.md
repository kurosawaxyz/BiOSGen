# Clinical Neural Style Transfer for Tumor Staining with Stable Diffusion Models
## Project structure
```txt
BiOSGen/
│── preprocess/            
│   ├── __init__.py      
│   ├── dataloader.py            
│   ├── patches_utils.py    
│   ├── tissue_mask.py      
│── osgen/                   
│   ├── __init__.py             
│   ├── base.py
│   ├── embeddings.py
│   ├── loss.py
│   ├── nn.py
│   ├── pipeline.py         # Main pipeline for OSGen
│   ├── unet.py
│   ├── utils.py
│   ├── vae.py
│── configs/               
│   ├── config.yml          
│── bin/                
│   ├── train.py  
│   ├── test.py           
│   ├── eval.py     
│── scripts/   
│   ├── batch_train.sh
│   ├── data_installer.py
│   ├── data_installer.sh
│   ├── setup_env.sh
│── demo/
│── docs/
│── assets/  
│── requirements.txt
│── README.md    
│── LICENSE
│── setup.py
│── pyproject.toml
```

## Users manual

### Quick setup
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

#### Important notice
**Warning**:

1. Environment setup using YAML file removed due to issues with `conda` and `pip` packages. *(Update on 2025-04-26)*

2. Severe issues may arise while building wheels for `flash-attn` due to incomppatibility with Python version >= 3.10. If you encounter this issue, please downgrade your Python version to 3.9 or 3.10.
```shell
# Create a new conda environment with Python 3.10
conda create -n biosgen python=3.10 -y

# Activate the environment
conda activate biosgen

# Check Python version
python --version

# Optional: upgrade pip
pip install --upgrade pip

# Install build tools (needed for compiling flash-attn)
pip install ninja packaging

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Now install flash-attn (will compile with correct CUDA/PyTorch setup)
pip install flash-attn --no-build-isolation -v

# Install other dependencies
pip install -r requirements.txt
conda install -c conda-forge python-graphviz -y
```

>*Note*: `-y` flag is used to automatically confirm the installation of packages without prompting for user input. This is useful when you want to install multiple packages in a single command without having to manually confirm each one.


### Data installation

#### Install train-test data

```bash
chmod +x scripts/data_installer.sh
./scripts/data_installer.sh
```

>**Note**: Data installation instruction for the EMPaCT dataset provided by [AI4SCR](https://github.com/AI4SCR/VirtualMultiplexer)
#### Downloading the EMPaCT dataset 

European Multicenter Prostate Cancer Clinical and Translational Research Group (EMPaCT) [[1](https://www.sciencedirect.com/science/article/pii/S0022534712029710), [2](https://www.sciencedirect.com/science/article/pii/S2405456917300020), [3](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2020.00246/full)] contains prostate cancer tissue microarrays (TMAs) from 210 patients with 4 cores per patient for several clinically relevant markers.

All images from Hematoxylin & Eosin (H&E) and Immunohistochemistry (IHC) stainings are uploaded to Zenodo under a Creative Commons Attribution 4.0 International License and can be dowloaded from this [link](https://zenodo.org/records/10066853).

#### Downloading Masks, Bounding Boxes and Data Splits

AI4SCR uploaded the all relevant information to [Figshare]( https://figshare.com/projects/VirtualMultiplexer/230498) and 
provide a notebook to demonstrate how to process and plot the data.