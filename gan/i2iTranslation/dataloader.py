from typing import Optional, Callable, Any, List, Tuple, Dict
import os
import random
from tqdm import tqdm
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch.transforms as transforms

class BaseDataset(Dataset):
    def __init__(self, args, image_fmt: str="png", **kwargs) -> None: 
        self.is_train = args.is_train
        self.patch_size = args.train.data.patch_size
        self.patch_norm = args.train.params.patch_norm
        self.max_src_samples = args.train.data.max_src_samples
        self.downsample = args.train.data.downsample
        
        self.downsampled_size = (
            int(self.patch_size // self.downsample),
            int(self.patch_size // self.downsample),
        )
        self.mode = 'train' if self.is_train else 'test'