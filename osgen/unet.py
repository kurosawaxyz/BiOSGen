# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from osgen.base import BaseModel
from osgen.embeddings import StyleExtractor, PositionalEmbedding
from osgen.nn import * 
from osgen.utils import Utilities
from osgen.vae import VanillaEncoder, VanillaDecoder


class UNet(BaseModel):
    pass
    # """
    # U-Net module for BiOSGen.
    # Source code: https://github.com/openai/consistency_models/blob/main/cm/unet.py#L518
    # """
    # def __init__(
    #     self,
    #     image_size,
    #     in_channels,
    #     model_channels,
    #     out_channels,
    #     num_res_blocks,
    #     attention_resolutions,
    #     dropout=0,
    #     channel_mult=(1, 2, 4, 8),
    #     conv_resample=True,
    #     dims=2,
    #     num_classes=None,
    #     use_checkpoint=False,
    #     use_fp16=False,
    #     num_heads=1,
    #     num_head_channels=-1,
    #     num_heads_upsample=-1,
    #     use_scale_shift_norm=False,
    #     resblock_updown=False,
    #     use_new_attention_order=False,
    # ):
    #     super().__init__()

    #     if num_heads_upsample == -1:
    #         num_heads_upsample = num_heads

    #     self.image_size = image_size
    #     self.in_channels = in_channels
    #     self.model_channels = model_channels
    #     self.out_channels = out_channels
    #     self.num_res_blocks = num_res_blocks
    #     self.attention_resolutions = attention_resolutions
    #     self.dropout = dropout
    #     self.channel_mult = channel_mult
    #     self.conv_resample = conv_resample
    #     self.num_classes = num_classes
    #     self.use_checkpoint = use_checkpoint
    #     self.dtype = th.float16 if use_fp16 else th.float32
    #     self.num_heads = num_heads
    #     self.num_head_channels = num_head_channels
    #     self.num_heads_upsample = num_heads_upsample

    #     time_embed_dim = model_channels * 4
    #     self.time_embed = nn.Sequential(
    #         nn.Linear(model_channels, time_embed_dim),
    #         nn.SiLU(),
    #         nn.Linear(time_embed_dim, time_embed_dim),
    #     )

    #     if self.num_classes is not None:
    #         self.label_emb = nn.Embedding(num_classes, time_embed_dim)

    #     ch = input_ch = int(channel_mult[0] * model_channels)
    #     self.input_blocks = nn.ModuleList(
    #         [TimestepEmbedSequential(nn.Conv2d(dims, in_channels, ch, 3, padding=1))]
    #     )
    #     self._feature_size = ch
    #     input_block_chans = [ch]
    #     ds = 1
    #     for level, mult in enumerate(channel_mult):
    #         for _ in range(num_res_blocks):
    #             layers = [
    #                 ResBlock(
    #                     ch,
    #                     time_embed_dim,
    #                     dropout,
    #                     out_channels=int(mult * model_channels),
    #                     dims=dims,
    #                     use_checkpoint=use_checkpoint,
    #                     use_scale_shift_norm=use_scale_shift_norm,
    #                 )
    #             ]
    #             ch = int(mult * model_channels)
    #             if ds in attention_resolutions:
    #                 layers.append(
    #                     AttentionBlock(
    #                         ch,
    #                         use_checkpoint=use_checkpoint,
    #                         num_heads=num_heads,
    #                         num_head_channels=num_head_channels,
    #                         use_new_attention_order=use_new_attention_order,
    #                     )
    #                 )
    #             self.input_blocks.append(TimestepEmbedSequential(*layers))
    #             self._feature_size += ch
    #             input_block_chans.append(ch)
    #         if level != len(channel_mult) - 1:
    #             out_ch = ch
    #             self.input_blocks.append(
    #                 TimestepEmbedSequential(
    #                     ResBlock(
    #                         ch,
    #                         time_embed_dim,
    #                         dropout,
    #                         out_channels=out_ch,
    #                         dims=dims,
    #                         use_checkpoint=use_checkpoint,
    #                         use_scale_shift_norm=use_scale_shift_norm,
    #                         down=True,
    #                     )
    #                     if resblock_updown
    #                     else Downsample(
    #                         ch, conv_resample, dims=dims, out_channels=out_ch
    #                     )
    #                 )
    #             )
    #             ch = out_ch
    #             input_block_chans.append(ch)
    #             ds *= 2
    #             self._feature_size += ch

    #     self.middle_block = TimestepEmbedSequential(
    #         ResBlock(
    #             ch,
    #             time_embed_dim,
    #             dropout,
    #             dims=dims,
    #             use_checkpoint=use_checkpoint,
    #             use_scale_shift_norm=use_scale_shift_norm,
    #         ),
    #         AttentionBlock(
    #             ch,
    #             use_checkpoint=use_checkpoint,
    #             num_heads=num_heads,
    #             num_head_channels=num_head_channels,
    #             use_new_attention_order=use_new_attention_order,
    #         ),
    #         ResBlock(
    #             ch,
    #             time_embed_dim,
    #             dropout,
    #             dims=dims,
    #             use_checkpoint=use_checkpoint,
    #             use_scale_shift_norm=use_scale_shift_norm,
    #         ),
    #     )
    #     self._feature_size += ch

    #     self.output_blocks = nn.ModuleList([])
    #     for level, mult in list(enumerate(channel_mult))[::-1]:
    #         for i in range(num_res_blocks + 1):
    #             ich = input_block_chans.pop()
    #             layers = [
    #                 ResBlock(
    #                     ch + ich,
    #                     time_embed_dim,
    #                     dropout,
    #                     out_channels=int(model_channels * mult),
    #                     dims=dims,
    #                     use_checkpoint=use_checkpoint,
    #                     use_scale_shift_norm=use_scale_shift_norm,
    #                 )
    #             ]
    #             ch = int(model_channels * mult)
    #             if ds in attention_resolutions:
    #                 layers.append(
    #                     AttentionBlock(
    #                         ch,
    #                         use_checkpoint=use_checkpoint,
    #                         num_heads=num_heads_upsample,
    #                         num_head_channels=num_head_channels,
    #                         use_new_attention_order=use_new_attention_order,
    #                     )
    #                 )
    #             if level and i == num_res_blocks:
    #                 out_ch = ch
    #                 layers.append(
    #                     ResBlock(
    #                         ch,
    #                         time_embed_dim,
    #                         dropout,
    #                         out_channels=out_ch,
    #                         dims=dims,
    #                         use_checkpoint=use_checkpoint,
    #                         use_scale_shift_norm=use_scale_shift_norm,
    #                         up=True,
    #                     )
    #                     if resblock_updown
    #                     else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
    #                 )
    #                 ds //= 2
    #             self.output_blocks.append(TimestepEmbedSequential(*layers))
    #             self._feature_size += ch

    #     self.out = nn.Sequential(
    #         normalization(ch),
    #         nn.SiLU(),
    #         zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
    #     )

    # def forward(self, x, timesteps, y=None):
    #     """
    #     Apply the model to an input batch.

    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     assert (y is not None) == (
    #         self.num_classes is not None
    #     ), "must specify y if and only if the model is class-conditional"

    #     hs = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)

    #     h = x.type(self.dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb)
    #         hs.append(h)
    #     h = self.middle_block(h, emb)
    #     for module in self.output_blocks:
    #         h = th.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
    #     h = h.type(x.dtype)
    #     return self.out(h)