import torch
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List

class DotDict:
    """Recursively convert a nested dictionary into an object supporting dot notation."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)  # Recursively convert sub-dictionaries
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return str(self.__dict__)


def load_config(config_path):
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)  # Load as dictionary
    return DotDict(args)  # Convert to dot notation

def load_data(data_csv_path, batch_size):
    data = pd.read_csv(data_csv_path)[:batch_size]
    return data

def preprocess_image(image_path, device):
    img = np.array(Image.open(image_path))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return img

def preprocess_array(image_array, device):
    img = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return img

def delete_tensor_gpu(tensor_dict: Dict):
    for k, v in tensor_dict.items():
        del v
