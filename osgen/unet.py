# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from osgen.base import BaseModel
from osgen.nn import * 

class AdaINUNet(BaseModel):
    """
    Improved UNet using StyledResBlock for better performance.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        style_strength=1.0,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.base_channels = base_channels
        
        # Time embedding
        time_embed_dim = time_emb_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_embed_dim, device=self.device),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, device=self.device),
        )
        
        # Initial convolution
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, device=self.device)
        )
        
        # Track current channel dimension
        ch = base_channels
        input_block_channels = [ch]
        
        # Setup resolutions for attention layers
        self.attention_resolutions = attention_resolutions
        
        # Downsampling path
        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = []
                out_ch = base_channels * mult
                
                # Add ResBlock
                layers.append(
                    StyledResBlock(
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        in_channels=ch,
                        out_channels=out_ch,
                        use_scale_shift_norm=True,
                        style_strength=style_strength,
                        device=self.device,
                    )
                )
                
                ch = out_ch
                
                # Add attention if appropriate for this resolution
                if 2**(level+3) in attention_resolutions:
                    layers.append(
                        FlashSelfAttention(z_dim=ch, heads=8)
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(ch)
            
            # Add downsampling if not the last level
            if level != len(channel_mults) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, use_conv=True, device=self.device)
                    )
                )
                input_block_channels.append(ch)
        
        # Middle block
        self.middle_block = TimestepEmbedSequential(
            StyledResBlock(
                emb_channels=time_embed_dim,
                dropout=dropout,
                in_channels=ch,
                out_channels=ch,
                use_scale_shift_norm=True,
                style_strength=style_strength,
                device=self.device,
            ),
            FlashSelfAttention(z_dim=ch, heads=8),
            StyledResBlock(
                emb_channels=time_embed_dim,
                dropout=dropout,
                in_channels=ch,
                out_channels=ch,
                use_scale_shift_norm=True,
                style_strength=style_strength,
                device=self.device,
            ),
        )
        
        # Upsampling path
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = []
                
                # Concatenate with skip connection
                in_ch = ch + input_block_channels.pop()
                out_ch = base_channels * mult
                
                # Add ResBlock
                layers.append(
                    StyledResBlock(
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        use_scale_shift_norm=True,
                        style_strength=style_strength,
                        device=self.device,
                    )
                )
                
                ch = out_ch
                
                # Add attention if appropriate for this resolution
                if 2**(level+3) in attention_resolutions:
                    layers.append(
                        FlashSelfAttention(z_dim=ch, heads=8)
                    )
                
                # Add upsampling if not the last block and at the end of a resolution level
                if level != 0 and i == num_res_blocks:
                    layers.append(
                        Upsample(ch, use_conv=True, device=self.device)
                    )
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # Final output layer
        # self.output_layer = nn.Sequential(
        #     nn.Conv2d(ch, out_channels, kernel_size=3, padding=1, device=self.device)
        # )
    
    def forward(self, x, timesteps, style=None):
        """
        Forward pass through the UNet.
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            style: Style tensor [B, style_channels, H, W]
                  If None, won't apply style conditioning
            
        Returns:
            Output tensor of same shape as input
        """
        
        # Time embedding using our locally defined function
        emb = timestep_embedding(timesteps, self.time_emb_dim)
        emb = self.time_embed(emb)
        
        # Handle None style
        if style is None:
            # Create an empty tensor with the same batch size as x
            batch_size = x.shape[0]
            # Use a small channel dimension that will be properly handled by StyledResBlock
            style = torch.zeros(batch_size, 1, 1, 1, device=x.device, dtype=x.dtype)
        
        # Initial processing
        h = x
        hs = []
        
        # Downsampling
        for module in self.input_blocks:
            if isinstance(module, TimestepEmbedSequential):
                h = module(h, emb, style)
            else:
                h = module(h)
            hs.append(h)

        # plt.imshow(h[0, 0].detach().float().cpu().numpy(), cmap='viridis')
        # plt.show()
        
        # Middle
        h = self.middle_block(h, emb, style)
        # plt.imshow(h[0, 0].detach().float().cpu().numpy(), cmap='viridis')
        # plt.show()
        
        # Upsampling
        for module in self.output_blocks:
            # print("hs: ", len(hs))
            # Concatenate with skip connection
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, style)
            # plt.imshow(h[0, 0].detach().float().cpu().numpy(), cmap='viridis')
            # plt.show()
        
        # Final output
        # h = self.output_layer(h)
        # plt.imshow(h[0, 0].detach().float().cpu().numpy(), cmap='viridis')
        # plt.show()
        return h