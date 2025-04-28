# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

from osgen.base import BaseModel
from osgen.embeddings import StyleExtractor, PositionalEmbedding
from osgen.nn import * 
from osgen.unet import UNetModel
from osgen.utils import Utilities
from osgen.vae import VanillaEncoder, VanillaDecoder

# Define BiOSGen main pipeline class
class Pipeline(BaseModel):
    pass