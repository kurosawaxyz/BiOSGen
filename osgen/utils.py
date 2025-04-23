# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import numpy as np
import torch

class Utilities:
    """
    A class containing utility functions for model operations.
    """
    def __init__(self):
        pass

    @staticmethod
    def load_model(model, path: str) -> None:
        """
        Load a model from a specified path.
        """
        model.load_state_dict(torch.load(path))
        model.eval()
    
    @staticmethod
    def convert_numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy array to a PyTorch tensor.
        """
        return torch.from_numpy(array).float().permute(2,0,1).unsqueeze(0)

    @staticmethod
    def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy array.
        """
        return tensor.cpu().detach().numpy()
    
    @staticmethod
    def convert_to_float32(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to float32.
        """
        return tensor.to(torch.float32)