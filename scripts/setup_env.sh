#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

conda create -n biosgen python=3.10 -y

# Activate the environment
conda activate biosgen

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