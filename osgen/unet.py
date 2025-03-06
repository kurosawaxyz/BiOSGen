import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora

from .network import CrossAttentionStyleFusion, TimestepEmbedSequential, ResBlock, Downsample, Upsample, timestep_embedding

class UNetModel(nn.Module):
    def __init__(
        self, 
        out_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.5,
        in_channels: int = 4,
        image_size: int = 32,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_classes=None,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        *args,
        **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.in_channels = in_channels
        self.image_size = image_size
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.num_classes = num_classes
        self.dtype = torch.float32  # Explicitly set dtype

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            lora.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            lora.Linear(time_embed_dim, time_embed_dim),
        )

        # Fix for label embedding
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.label_emb = None
        
        ch = input_ch = int(channel_mult[0] * model_channels)
        
        # Input block
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            nn.ModuleList([
                lora.Conv2d(in_channels, ch, kernel_size=3, padding=1)
            ])
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = nn.ModuleList([
                    ResBlock(
                        emb_channels=time_embed_dim,
                        in_channels=ch,
                        dropout=dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ])
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        CrossAttentionStyleFusion(
                            latent_channels=ch,
                            cond_dim=time_embed_dim,
                        )
                    )
                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                out_ch = ch
                downsample_block = nn.ModuleList()
                if resblock_updown:
                    downsample_block.append(
                        ResBlock(
                            emb_channels=time_embed_dim,
                            in_channels=ch,
                            dropout=dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                    )
                else:
                    downsample_block.append(
                        Downsample(
                            ch,
                            conv_resample,
                            out_channels=out_ch,
                        )
                    )
                
                self.input_blocks.append(downsample_block)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            emb_channels=time_embed_dim,
                            in_channels=ch,
                            dropout=dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            out_channels=out_ch,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                emb_channels=time_embed_dim,
                in_channels=ch,
                dropout=dropout,
                out_channels=ch,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
                
            CrossAttentionStyleFusion(
                latent_channels=ch,
                cond_dim=time_embed_dim,
            ),
            ResBlock(
                emb_channels=time_embed_dim,
                in_channels=ch,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        emb_channels=time_embed_dim,
                        in_channels = ch + ich,
                        dropout=dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        CrossAttentionStyleFusion(
                            latent_channels=ch,
                            cond_dim=time_embed_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            in_channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            out_channels=out_ch,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.SiLU(),
            lora.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            for layer in module:  # Iterate through each layer inside the module
                print(f"Layer type: {type(layer).__name__}")
                if type(layer).__name__ in ["ResBlock", "CrossAttentionStyleFusion"]:
                    h = layer(h, emb)
                else:
                    h = layer(h)      # Remove emb for now
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:  # Iterate through each layer inside the module
                print(f"Layer type: {type(layer).__name__}")
                if type(layer).__name__ in ["ResBlock", "CrossAttentionStyleFusion"]:
                    h = layer(h, emb)
                else:
                    h = layer(h)      # Remove emb for now
        h = h.type(x.dtype)
        return self.out(h)