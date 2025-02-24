import torch.nn as nn
from i2iTranslation.vgg import *
import torch

class TileLevelCriterion(nn.Module):
    """ Code inspired from https://github.com/AI4SCR/VirtualMultiplexer/blob/master/i2iTranslation/models/global_objectives.py """
    # https://www.biorxiv.org/content/10.1101/2023.11.29.568996v1.full.pdf - Page 25
    
    def __init__(self, args):
        super(TileLevelCriterion, self).__init__()  
        self.args = args
        self.device = args.device  # Access device properly
        self.prepare_model()  

    def prepare_model(self):
        image_model_name = self.args.train.params.image_model_name  
        if image_model_name == 'vgg16':
            model = Vgg16(requires_grad=False)
        elif image_model_name == 'vgg19':
            model = Vgg19(requires_grad=False)
        elif image_model_name == 'vgg16experimental':
            model = Vgg16Experimental(requires_grad=False)
        else:
            raise ValueError(f'{image_model_name} not supported.')

        self.content_layer_names = model.content_layer_names
        self.style_layer_names = model.style_layer_names
        self.model = model.to(self.device).eval()  #

    def content_loss(self, real_content, fake_content):
        real_content = real_content.detach()
        return nn.MSELoss(reduction='mean')(real_content, fake_content)

    def style_loss(self, real_style, fake_style, weighted=True):
        real_style = real_style.detach()     # we dont need the gradient of the target
        size = real_style.size()

        if not weighted:
            weights = torch.ones(size=real_style.shape[0])
        else:
            # https://arxiv.org/pdf/2104.10064.pdf
            Nl = size[1] * size[2]  # C x C = C^2
            real_style_norm = torch.linalg.norm(real_style, dim=(1, 2))
            fake_style_norm = torch.linalg.norm(fake_style, dim=(1, 2))
            normalize_term = torch.square(real_style_norm) + torch.square(fake_style_norm)
            weights = Nl / normalize_term

        se = (real_style.view(size[0], -1) - fake_style.view(size[0], -1)) ** 2
        return (se.mean(dim=1) * weights).mean()

    def forward(self, content_img, style_img, fake_img):
        content_img_feature_maps = self.model(content_img)
        style_img_feature_maps = self.model(style_img)
        fake_img_feature_maps = self.model(fake_img)

        real_content_representation = [x for cnt, x in enumerate(content_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        real_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        fake_content_representation = [x for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        fake_style_representation = [gram_matrix(x) for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        # content loss
        content_loss = 0
        for i, layer in enumerate(self.content_layer_names):
            content_loss += self.content_loss(real_content_representation[i], fake_content_representation[i])

        # style loss
        style_loss = 0
        for i, layer in enumerate(self.style_layer_names):
            style_loss += self.style_loss(real_style_representation[i], fake_style_representation[i])

        return content_loss, style_loss