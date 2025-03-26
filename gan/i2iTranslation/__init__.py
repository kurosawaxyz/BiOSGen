from .base_model import BaseModel
from .global_loss import TileLevelCriterion
from .local_loss import GANLoss, PatchNCELoss
from .model import i2iTranslationModel
from .network import * 
from .utils import DotDict
from .vgg import Vgg16, Vgg19, Vgg16Experimental


def create_model(args, device):
    """Create model for training
    """
    model = i2iTranslationModel(args)
    model.setup(args)
    return model