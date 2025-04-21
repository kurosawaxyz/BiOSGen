# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

from osgen.base import BaseModel

class Encoder(BaseModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # Define the encoder model here
        # For example, you can use a pre-trained model from torchvision.models
        # self.model = torchvision.models.resnet50(pretrained=True)
        # self.model.fc = nn.Identity()  # Remove the final classification layer
        
    #TODO: positional encoding + transformer encoder !