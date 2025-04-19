import torch 
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM 

import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
import matplotlib.patches as patches

from .tissue_mask import GaussianTissueMask

# moondream
def load_md(
    device: str = "cuda"
):
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
    ).to(device)
    return model

def describe_img(
    image_path: str,
    device: str = "cuda"
):
    img = Image.open(image_path)
    model = load_md(device)
    enc_image = model.encode_image(img)
    print(model.query(enc_image, "Describe this image.\n"))

def read_image(
    image_fpath: str
) -> np.ndarray:
    return np.array(Image.open(image_fpath))