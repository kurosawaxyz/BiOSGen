#!/bin/bash

# Only execute with CUDA, impossible on Mac M1 GPU

# Load data
python -m cyclegan.loader --config_path configs/config.yml --style_path demo/img/A6_TMA_15_02_IVB_NKX.png --original_path demo/img/A4_TMA_15_02_IVB_HE.png --dataset_path cyclegan/data/ && \
echo "Data loaded."

# Check if the repository already exists
if [ ! -d "pytorch-CycleGAN-and-pix2pix" ]; then
  # Clone the repository if it doesn't exist
  git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git && \
  echo "Repository cloned."
else
  echo "Repository already exists, skipping git clone."
fi

# Change directory to the cloned repository
cd pytorch-CycleGAN-and-pix2pix || exit

# Install required dependencies
pip install -r requirements.txt && \
echo "Dependencies installed."

# Train the model
python train.py \
  --dataroot ../cyclegan/data/ \
  --name tumor_cyclegan \
  --model cycle_gan \
  --display_id -1 && \
echo "Training started."


