#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

styles=("p53" "CD146" "NKX3" "ERG" "AR" "CD44")

for style in "${styles[@]}"; do
  python bin/train.py \
    --config configs/train_config.yml \
    --original HE \
    --style "$style" \
    --checkpoints checkpoints \
    --data data
done
