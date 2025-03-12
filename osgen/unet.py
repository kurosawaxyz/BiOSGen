import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora

from .nn import TimestepEmbedSequential, ResBlock, Downsample, Upsample, timestep_embedding

class UNetModel(nn.Module):
    def __init__(
        self, 
        out_channels,
        model_channels,
        num_res_blocks,
        dropout=0.5,
        in_channels: int = 4,
        image_size: int = 128,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_classes=None,
        channel_mult=(1, 2, 4, 8),  # Ensure minimum downsampled size is 16x16
        use_conv=True,
        is_trainable: bool = True,
        lora_rank: int = 8,
        *args,
        **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.in_channels = in_channels
        self.image_size = image_size
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            lora.Linear(model_channels, time_embed_dim, r = lora_rank),
            nn.SiLU(),
            lora.Linear(time_embed_dim, time_embed_dim, r = lora_rank),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(nn.ModuleList([lora.Conv2d(in_channels, ch, kernel_size=3, padding=1)]))
        input_block_chans = [ch]
        
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        emb_channels=time_embed_dim,
                        in_channels=ch,
                        dropout=dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                        is_trainable=is_trainable,
                        lora_rank=lora_rank,
                        use_conv=use_conv  # Keep original use_conv for downsampling path
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                downsample_block = nn.ModuleList([
                    Downsample(ch, use_conv, out_channels=ch, is_trainable=is_trainable, lora_rank=lora_rank)
                ])
                self.input_blocks.append(downsample_block)
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                emb_channels=time_embed_dim,
                in_channels=ch,
                dropout=dropout,
                out_channels=ch,
                use_scale_shift_norm=use_scale_shift_norm,
                is_trainable=is_trainable,
                lora_rank=lora_rank,
                use_conv=use_conv  # Set to use_conv (not negated) for middle block
            )
        )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        emb_channels=time_embed_dim,
                        in_channels=ch + ich,
                        dropout=dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                        is_trainable=is_trainable,
                        lora_rank=lora_rank,
                        use_conv=use_conv  # Set to use_conv (not negated) for upsampling path
                    )
                ]
                ch = int(model_channels * mult)

                # Only upsample at specific levels to reach 128×128
                if level != 0 and i == num_res_blocks:  # Restore the condition i == num_res_blocks
                    # Control upsampling to ensure we reach 128×128
                    # We want to upsample exactly enough to reach our target size
                    layers.append(Upsample(ch, use_conv, out_channels=ch, is_trainable=is_trainable, lora_rank=lora_rank))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Remove the final_upsample to prevent over-upsampling
        # self.final_upsample = Upsample(ch, not use_conv, out_channels=ch, is_trainable=is_trainable, lora_rank=lora_rank)
        
        self.out = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.SiLU(),
            lora.Conv2d(ch, out_channels, kernel_size=3, padding=1, r=lora_rank),
        )

        if is_trainable:
            lora.mark_only_lora_as_trainable(self, bias='lora_only')
            # print("UNetModel is trainable", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (self.num_classes is not None)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        h = x
        for module in self.input_blocks:
            for layer in module:
                h = layer(h) if not isinstance(layer, ResBlock) else layer(h, emb)
                # print(f"After {layer.__class__.__name__}: {h.shape}")
            hs.append(h)
        
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            skip = hs.pop()

            # Ensure batch size matches
            if skip.shape[0] != h.shape[0]:
                skip = skip.expand(h.shape[0], -1, -1, -1)  # Expand batch dimension if needed

            # Ensure spatial size matches
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")

            # print(f"Upsampling: h={h.shape}, skip={skip.shape}")  # Debugging output
            h = torch.cat([h, skip], dim=1)
            
            for layer in module:
                h = layer(h) if not isinstance(layer, ResBlock) else layer(h, emb)

        # Remove this line since we removed the final_upsample
        # h = self.final_upsample(h)
        
        return self.out(h)
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)