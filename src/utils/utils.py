import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu

tf_to_grayscale = transforms.Grayscale(num_output_channels=1)
tf_to_PIL = transforms.ToPILImage()


def mask_generator(
    shadow_img: torch.Tensor, shadow_free_img: torch.Tensor
) -> torch.Tensor:
    """
    generate mask image from shadow and shadow free image
    """
    image_free = tf_to_grayscale(
        tf_to_PIL(((shadow_free_img.data.squeeze(0) + 1) * 0.5).cpu())
    )
    image_shadow = tf_to_grayscale(
        tf_to_PIL(((shadow_img.data.squeeze(0) + 1) * 0.5).cpu())
    )

    diff = np.asarray(image_free, dtype="float32") - np.asarray(
        image_shadow, dtype="float32"
    )  # difference between shadow image and shadow_free image

    L = threshold_otsu(diff)
    mask = (
        torch.tensor((np.float32(diff >= L) - 0.5) / 0.5)
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )  # -1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    # print("MASK type (mask_generator:)", type(mask))
    return mask


# pass


def weights_init(model: nn.Module) -> None:
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
    """
    Computes a multiplicative factor given an integer parameter epoch,
    one for each group in optimizer.param_groups
    """

    def __init__(self, num_epochs: int, offset: int, decay_start_epoch: int):
        assert (num_epochs - decay_start_epoch) > 0
        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int):
        # TODO description
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.num_epochs - self.decay_start_epoch
        )


# TODO description everywhere
class QueueMask:
    def __init__(self, lenght: int) -> None:
        self.max_len = lenght
        self.queue = []

    def insert(self, mask):
        print("insert works")
        print("Mask:", type(mask))
        if self.queue.__len__() >= self.max_len:
            self.queue.pop(0)
        self.queue.append(mask)
        print("q len: ", len(self.queue))

    def rand_item(self):
        assert len(self.queue) > 0
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert len(self.queue) > 0, "Error! Empty queue!"
        return self.queue[self.queue.__len__() - 1]


class Buffer:
    """
    Used for temporary storage of shadow masks and shadowed images.
    It is initialized with maximum size and to store a mask or a shadow
    push_and_pop function can be utilised.
    """

    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        res = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                res.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    res.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    res.append(element)
        return Variable(torch.cat(res))
