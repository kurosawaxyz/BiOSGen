#!/bin/bash

python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/HE_zenodo.zip?download=1 --output /root/BiOSGen/data/HE.zip
python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/NKX3_zenodo.zip?download=1 --output /root/BiOSGen/data/NKX3.zip
python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023048 --output /root/BiOSGen/data/tissue_mask.zip
python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023045 --output /root/BiOSGen/data/bbox.zip
