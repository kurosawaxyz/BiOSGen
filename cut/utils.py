import torch
import pandas as pd
import numpy as np
from PIL import Image

def load_data(data_csv_path, batch_size):
    data = pd.read_csv(data_csv_path)[:batch_size]
    return data

def preprocess_image(image_path, device):
    img = np.array(Image.open(image_path))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return img

