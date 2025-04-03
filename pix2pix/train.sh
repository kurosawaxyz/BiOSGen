#!/bin/bash

# Only execute with CUDA, impossible on Mac M1 GPU

# Load data
python -m pix2pix.loader --config_path configs/config.yml --style_path demo/img/A6_TMA_15_02_IVB_NKX.png --original_path demo/img/A4_TMA_15_02_IVB_HE.png --dataset_path pix2pix/data/ && \
echo "Data loaded."

# Check if the repository already exists
if [ -d "pytorch-CycleGAN-and-pix2pix" ]; then
  # Remove the existing repository
  echo "Repository already exists, removing it..."
  rm -rf pytorch-CycleGAN-and-pix2pix
fi

# Clone the repository
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git && \
echo "Repository cloned."

# Change directory to the cloned repository
cd pytorch-CycleGAN-and-pix2pix || exit

# Install required dependencies
pip install -r requirements.txt && \
echo "Dependencies installed."

# Train the model
python train.py \
  --dataroot ../pix2pix/data/ \
  --name tumor_pix2pix \
  --model pix2pix \
  --display_id -1 && \
echo "Training started."