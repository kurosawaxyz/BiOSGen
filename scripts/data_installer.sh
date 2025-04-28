#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

mkdir -p data

python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/HE_zenodo.zip?download=1 --output /root/BiOSGen/data/HE.zip
python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/NKX3_zenodo.zip?download=1 --output /root/BiOSGen/data/NKX3.zip
python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023048 --output /root/BiOSGen/data/tissue_mask.zip
python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023045 --output /root/BiOSGen/data/bbox.zip

# Check if unzip command is available
if ! command -v unzip &> /dev/null
then
    apt install unzip
fi

# Unzip the downloaded files
unzip data/HE.zip -d data/HE
unzip data/NKX3.zip -d data/NKX3
unzip data/tissue_mask.zip -d data/tissue_mask
unzip data/bbox.zip -d data/bbox

# Remove the zip files after extraction
rm data/HE.zip
rm data/NKX3.zip
rm data/tissue_mask.zip
rm data/bbox.zip