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
        image_size: int = 256,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_classes=None,
        channel_mult=(1, 2, 4, 8),  # Ensure minimum downsampled size is 16x16
        use_conv=True,
        is_trainable: bool = True,
        lora_rank: int = 16,
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
            lora.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            lora.Linear(time_embed_dim, time_embed_dim),
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
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                downsample_block = nn.ModuleList([
                    Downsample(ch, use_conv, out_channels=ch)
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
            ),
            # ResBlock(
            #     emb_channels=time_embed_dim,
            #     in_channels=ch,
            #     dropout=dropout,
            #     out_channels=ch,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
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
                        use_conv=True
                    )
                ]
                ch = int(model_channels * mult)

                # Ensure we upsample back to 128×128
                if level != 0 and (ch < self.model_channels * max(channel_mult)):   # Remove condition i == num_res_blocks 
                    layers.append(Upsample(ch, use_conv, out_channels=ch))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Ensure final resolution is 128×128
        self.final_upsample = Upsample(ch, use_conv, out_channels=ch)

        
        self.out = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.SiLU(),
            lora.Conv2d(ch, out_channels, kernel_size=3, padding=1),
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

        
        return self.out(h)
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)