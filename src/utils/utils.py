from pydoc import classname
from re import L
import torch
import torch.nn as nn
from PIL import Image


def mask_generator(shadow_img, shadow_free_img) -> torch.Tensor:
    """
    generate mask image from shadow and shadow free image
    """
    pass


def weights_init(model):
    """
    Function takes an initialized model as input and reinitializes all convolutional,
    convolutional-transpose, and batch normalization layers to meet this criteria.
    """
    classname: str = model.__class__.__name__
    if classname.find("Conv") != -1:
        # it's done when there isn't any looked for argument in classname
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class LR_lambda:
    def __init__(self, num_epochs, offset, decay_start_epoch):
        assert (num_epochs - decay_start_epoch) > 0
        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.num_epochs - self.decay_start_epoch
        )
