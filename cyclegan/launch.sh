#!/bin/bash

git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git

cd pytorch-CycleGAN-and-pix2pix

pip install -r requirements.txt

# Load data
python -m cyclegan.loader \
  --config_path /content/BiOSGen/configs/config.yml \
  --style_path /content/BiOSGen/demo/img/A6_TMA_15_02_IVB_NKX.png \
  --original_path /content/BiOSGen/demo/img/A4_TMA_15_02_IVB_HE.png \
  --dataset_path /content/BiOSGen/cyclegan

# Train
python train.py \
  --dataroot /content/BiOSGen/cyclegan \
  --name tumor \
  --model cycle_gan \
  --display_id -1
