#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

mkdir -p data

python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/HE_zenodo.zip?download=1 --output data/HE.zip
python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/NKX3_zenodo.zip?download=1 --output data/NKX3.zip
python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023048 --output data/tissue_mask.zip
python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023045 --output data/bbox.zip

# Check if unzip command is available
if ! command -v unzip &> /dev/null
then
    apt install unzip
fi

# Unzip the downloaded files
unzip data/HE.zip -d data/
unzip data/NKX3.zip -d data/
unzip data/tissue_mask.zip -d data/
unzip data/bbox.zip -d data/

# Remove the zip files after extraction
rm data/HE.zip
rm data/NKX3.zip
rm data/tissue_mask.zip
rm data/bbox.zip