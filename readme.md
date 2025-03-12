# BiOSGen
This project presents the work conducted during my first-year master's internship at the ***Paris Brain Institute*** (ICM), in association with the ***DU Sorbonne Data Analytics*** program at ***Université Paris 1 Panthéon-Sorbonne***.

The primary objective was to develop multi-scale image preprocessing and analysis techniques, as well as an integration and fusion method for image analysis. Specifically, this work focused on tumor staining style transfer from H&E to IHC images, aiming to reduce financial costs and save time for more technical analyses.

Two style transfer models were compared: one using a traditional GAN-based approach and another leveraging a more advanced diffusion-based architecture.

Future work will focus on refining the diffusion model to generate more accurate IHC images while incorporating patient-specific health factors that may influence tumor growth, progression, and treatment response.

## Model architecture
![](assets/main%20diagram.png)

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
│── data/                   # hidden directory       
│── assets/                 # hidden directory
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
```
#### For miniforge users:
**Warning**:

You will need to remove line 4 and line 32-34 from `environment.yml` to avoid conflicts while setting up the environment using YAML file. 

As you can't access to several `conda` packages in miniforge, you will not be enable to use Moondream model from *vikhyatk/moondream2*. However, there are alternatives such as *HuggingFaceTB/SmolVLM-Instruct*, *microsoft/OmniParser-v2.0*, etc. 

```bash
conda env create -f environment.yml -k
```

## Data installation
