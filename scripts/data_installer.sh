#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

mkdir -p data

# Declare an associative array of filenames and URLs
declare -A files=(
  ["HE"]="https://zenodo.org/records/10066853/files/HE_zenodo.zip?download=1"
  ["AR"]="https://zenodo.org/records/10066853/files/AR_zenodo.zip?download=1"
  ["CD146"]="https://zenodo.org/records/10066853/files/CD146_zenodo_part1.zip?download=1"
  ["CD44"]="https://zenodo.org/records/10066853/files/CD44_zenodo.zip?download=1"
  ["ERG"]="https://zenodo.org/records/10066853/files/ERG_zenodo.zip?download=1"
  ["p53"]="https://zenodo.org/records/10066853/files/p53_zenodo.zip?download=1"
  ["NKX3"]="https://zenodo.org/records/10066853/files/NKX3_zenodo.zip?download=1"
  ["tissue_mask"]="https://figshare.com/ndownloader/files/51023048"
  ["bbox"]="https://figshare.com/ndownloader/files/51023045"
)

# Download the files
for name in "${!files[@]}"; do
  python -m scripts.data_installer --url "${files[$name]}" --output "data/${name}.zip"
done

# Ensure unzip is installed
if ! command -v unzip &> /dev/null; then
  echo "Installing unzip..."
  apt update && apt install -y unzip
fi

# Unzip the files
for name in "${!files[@]}"; do
  unzip "data/${name}.zip" -d data/
done

# Remove zip files after extraction
for name in "${!files[@]}"; do
  rm "data/${name}.zip"
done
