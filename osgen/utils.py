# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Utilities:
    """
    A class containing utility functions for model operations.
    """
    def __init__(self):
        pass

    @staticmethod
    def train_test_split_indices(dataset, train_ratio: float = 0.8):
        """
        Split dataset indices into training and testing sets.

        Args:
            dataset: The dataset to split (only length is used).
            train_ratio: The proportion of the dataset to include in the training set.

        Returns:
            A tuple of (train_indices, test_indices).
        """
        n = len(dataset)
        indices = torch.randperm(n).tolist()
        n_train = int(n * train_ratio)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        return train_indices, test_indices

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
    def convert_to_bfloat16(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to bfloat16.
        """
        return tensor.to(torch.bfloat16)
    
    @staticmethod
    def convert_to_float32(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to float32.
        """
        return tensor.to(torch.float32)

    @staticmethod
    def visualize_attention(attention_weights, H, W, num_style_tokens):
        """
        Visualize attention weights as heatmaps.

        Args:
            attention_weights: Tensor of shape (B, heads, H*W, N)
            H, W: Height and width of the spatial dimensions (of z)
            num_style_tokens: Number of style tokens (H_style * W_style)
        """
        # Average over heads, take first batch
        avg_attention = attention_weights[0].mean(dim=0)  # (H*W, N)

        # Sanity check
        assert avg_attention.shape[0] == H * W, f"Expected {H*W} spatial positions, got {avg_attention.shape[0]}"
        assert avg_attention.shape[1] == num_style_tokens, f"Expected {num_style_tokens} style tokens, got {avg_attention.shape[1]}"

        # Create a figure for overall attention patterns
        plt.figure(figsize=(15, 10))
        
        # Plot average attention to each style token
        token_importance = avg_attention.mean(dim=0)  # (N,)
        plt.subplot(2, 1, 1)
        plt.bar(range(num_style_tokens), token_importance.cpu().numpy())
        plt.title('Average Attention per Style Token')
        plt.xlabel('Style Token Index')
        plt.ylabel('Average Attention')
        
        # Plot heatmap of spatial positions (flattened) attending to style tokens
        plt.subplot(2, 1, 2)
        plt.imshow(avg_attention.cpu().numpy(), aspect='auto', cmap='viridis')
        plt.title('Attention Heatmap: Spatial Positions (rows) vs Style Tokens (columns)')
        plt.xlabel('Style Token Index')
        plt.ylabel('Spatial Position (flattened HxW)')
        plt.colorbar(label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig('attention_overview.png')
        plt.close()

        # Now visualize attention per style token spatially
        num_tokens_to_show = min(6, num_style_tokens)  # Show up to 6 tokens
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(num_tokens_to_show):
            # For each token, get its attention map across spatial positions
            spatial_attn = avg_attention[:, i].reshape(H, W)  # (H, W)
            ax = axes[i]
            im = ax.imshow(spatial_attn.cpu().numpy(), cmap='viridis')
            ax.set_title(f'Style Token {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('spatial_attention_maps.png')
        plt.close()

        print("Visualization complete! Check 'attention_overview.png' and 'spatial_attention_maps1.png'")


    @staticmethod
    def convert_module_to_bf16(module):
        """
        Convert primitive modules to bfloat16.
        Supports Conv, Linear, InstanceNorm, LayerNorm and other common layers.
        """
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            module.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
        
        elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, 
                            nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, 
                            nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if module.weight is not None:
                module.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                module.running_mean = module.running_mean.to(torch.bfloat16)
            if hasattr(module, 'running_var') and module.running_var is not None:
                module.running_var = module.running_var.to(torch.bfloat16)

    @staticmethod
    def convert_model_to_bf16(model):
        """
        Convert an entire model to bfloat16.
        """
        for module in model.modules():
            Utilities.convert_module_to_bf16(module)
        
        return model

    @staticmethod
    def convert_module_to_f16(l):
        """
        Convert primitive modules to float16.
        """
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            l.weight.data = l.weight.data.to(torch.float16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.float16)

    @staticmethod
    def convert_model_to_f16(model):
        """
        Convert an entire model to float16.
        """
        for module in model.modules():
            Utilities.convert_module_to_f16(module)
        
        return model

    @staticmethod
    def convert_module_to_f32(l):
        """
        Convert primitive modules to float32, undoing convert_module_to_f16().
        """
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

    @staticmethod
    def convert_model_to_f32(model):
        """
        Convert an entire model to float32, undoing convert_model_to_f16().
        """
        for module in model.modules():
            Utilities.convert_module_to_f32(module)
        
        return model