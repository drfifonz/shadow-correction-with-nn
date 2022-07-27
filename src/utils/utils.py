from pydoc import classname
from re import L
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable


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


def allocate_memory(opt):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    input_shadow = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)
    input_mask = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)
    mask_non_shadow = Variable(
        Tensor(opt.batch_size, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False
    )
    return [input_shadow, input_mask, target_real, target_fake, mask_non_shadow]


class QueueMask:
    def __init__(self, lenght) -> None:
        self.max_len = lenght
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_len:
            self.queue.pop(0)
        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__ > 0, "Error! Empty queue!"
        return self.queue[self.queue.__len__() - 1]
