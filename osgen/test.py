import torch 
import torch.nn as nn

from osgen.nn import describe_img
from vae import VAEncoder, VAEDecoder
from vcm import extract_style_emb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


if __name__ == "__main__":
    IMAGE_PATH_SRC = "/Users/hoangthuyduongvu/Documents/icm/tumor-augmentation/data/HE/A3_TMA_15_02_IVB_HE.png"
    describe_img(IMAGE_PATH_SRC)

    IMAGE_PATH_DST = "/Users/hoangthuyduongvu/Documents/icm/tumor-augmentation/data/NKX3/A3_TMA_15_02_IB_NKX.png"
    describe_img(IMAGE_PATH_DST)

    print("Style Embedding Extraction:\n", extract_style_emb(IMAGE_PATH_SRC))
    print("Style Embedding Extraction:\n", extract_style_emb(IMAGE_PATH_DST))


    # Test encoder and decoder
    encoder = VAEncoder(input_dim=256, output_dim=128)
    decoder = VAEDecoder(input_dim=128, output_dim=256)