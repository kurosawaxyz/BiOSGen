# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
from torchvision import transforms
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="deep")
from torchviz import make_dot
import loralib as lora

class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    def __init__(
            self, 
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
            **kwargs
    ) -> None:
        super().__init__()
        self.device = device
        self.to(device)

    def check_dtype(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.dtype}")

    def count_dtypes(self):
        dtype_counts = {}
        for param in self.parameters():
            dtype = param.dtype
            if dtype in dtype_counts:
                dtype_counts[dtype] += 1
            else:
                dtype_counts[dtype] = 1
        
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} parameters")

    def get_model_size(self):
        size_bytes = 0
        for param in self.parameters():
            size_bytes += param.nelement() * param.element_size()
        return size_bytes / 1024**2  # Convert to MB

    def count_dtypes(self):
        dtype_counts = {}
        for param in self.parameters():
            dtype = param.dtype
            if dtype in dtype_counts:
                dtype_counts[dtype] += 1
            else:
                dtype_counts[dtype] = 1
        
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} parameters")

    def train(self, mode: bool = True) -> None:
        """
        Placeholder method for training the model.
        """
        lora.mark_only_lora_as_trainable(self, bias='lora_only')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        raise NotImplementedError("Forward method not implemented in base class.")
    
    def visualize_network(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Visualize the model architecture and parameters.
        """
        # Do not double convert tensor to device
        y = self(x)
        dot = make_dot(y, params=dict(list(self.named_parameters()) + [('x', x)]))
        dot.render(f"model_architecture_{self.__class__.__name__}", format="png")

    def get_model_summary(self) -> str:
        """
        Generate a summary of the model architecture and number of parameters.
        """
        model_summary = str(self)
        return model_summary
    
    def get_num_parameters(self) -> int:
        """
        Calculate the total number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def preview_visualize_feature_maps(
        self,
        image: Image.Image,
        num_maps: int = 6,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        size: int = 512,
        save_path: str = None,
        show: bool = False
    ) -> None:
        
        """
        Note: Do not use with Vanilla VAE, it will not work
        """

        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # Convert PIL image to tensor
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self(img_tensor)             # do not convert img_tensor to device, not working 

        feature_maps = features.squeeze(0).cpu().numpy()
        _, axs = plt.subplots(1, num_maps, figsize=(15, 5))
        for i in range(num_maps):
            axs[i].imshow(feature_maps[i], cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'Feature {i}')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)

        if show:
            plt.show()