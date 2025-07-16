# Clinical Neural Style Transfer for Tumor Staining with Stable Diffusion Models

## Model architecture
<div align="center">

  <img src="docs/assets/diagram.png" alt="architecture" width="750"/>

</div>

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


## How to use
### Training
```bash
chmod +x scripts/batch_train.sh
./scripts/batch_train.sh
```

```mermaid
flowchart TD
    A[Input: 50x3x512x512] --> B[OSGenPipeline]
    
    B --> C[StyleExtractor]
    B --> D[VanillaEncoder]
    B --> E[AdaINUNet]
    B --> F[VanillaDecoder]
    
    %% StyleExtractor branch
    C --> C1[Sequential CNN Backbone]
    C1 --> C1a[Conv2d + BatchNorm + ReLU]
    C1a --> C1b[MaxPool2d: 256x256→128x128]
    C1b --> C1c[ResNet Blocks]
    C1c --> C1d[2048 channels at 16x16]
    
    C --> C2[Conv2d: 512 channels]
    C --> C3[VGG-like Sequential]
    C3 --> C3a[Multiple Conv+ReLU blocks]
    C3a --> C3b[MaxPool operations]
    C3b --> C3c[256 channels at 64x64]
    
    C --> C4[Style Processing]
    C4 --> C4a[Conv2d + InstanceNorm]
    C4a --> C4b[Style Conv blocks]
    C4b --> C4c[Upsampling stages]
    C4c --> C5[Output: 64x128x128]
    
    %% VanillaEncoder branch
    D --> D1[Sequential Processing]
    D1 --> D1a[Conv blocks: 32→64 channels]
    D1a --> D1b[Downsampling: 256x256→128x128]
    D --> D2[Linear Layers]
    D2 --> D2a[Mean projection: 64 dims]
    D2 --> D2b[Logvar projection: 64 dims]
    D --> D3[Final Conv2d]
    D3 --> D4[Output: 64x128x128]
    
    %% AdaINUNet branch
    E --> E1[Time Embedding]
    E1 --> E1a[Linear: 512 dims]
    E1a --> E1b[SiLU activation]
    E1b --> E1c[Linear: 512 dims]
    
    E --> E2[Encoder Blocks]
    E2 --> E2a[Conv2d: 64 channels]
    E2a --> E2b[TimestepEmbedSequential blocks]
    E2b --> E2c[Downsampling: 128→64→32→16]
    E2c --> E2d[Channel expansion: 64→128→256→512]
    
    E --> E3[Middle Block]
    E3 --> E3a[StyledResBlock]
    E3a --> E3b[FlashSelfAttention]
    E3b --> E3c[StyledResBlock]
    
    E --> E4[Decoder Blocks]
    E4 --> E4a[Upsampling stages]
    E4a --> E4b[Skip connections]
    E4b --> E4c[Channel reduction: 512→256→128→64]
    E4c --> E5[Output: 64x128x128]
    
    %% VanillaDecoder branch
    F --> F1[Decoder Sequential]
    F1 --> F1a[ConvTranspose2d: 1024 channels]
    F1a --> F1b[InstanceNorm + ReLU]
    F1b --> F1c[ConvTranspose2d: 256 channels]
    F1c --> F1d[InstanceNorm + ReLU]
    F1d --> F1e[ConvTranspose2d: 3 channels]
    F1e --> F1f[Tanh activation]
    F1f --> F1g[Upsample to 512x512]
    F1g --> G[Final Output: 50x3x512x512]
    
    %% Data flow connections
    C5 -.-> E2
    D4 -.-> E2
    E5 -.-> F1
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef output fill:#f3e5f5
    classDef processing fill:#fff3e0
    classDef attention fill:#ffebee
    
    class A input
    class G output
    class C,D,E,F processing
    class E3b attention
```