import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def create_dataset(directory_path: str) -> list:
    """
    creating list of images
    """
    return list(sorted(os.listdir(directory_path)))


def image_loader(image_path: str):
    """
    loading image by pillow
    """
    # https://github.com/python-pillow/Pillow/issues/835
    return Image.open(image_path).convert("RGB")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, loader, transforms_=None, target_transform=None):

        # print("------------")
        # print(type(create_dataset(root)))
        # print(len(list(create_dataset(root))))
        # print("------------")
        images = create_dataset(root)
        # print(len(images))
        # if len(images == 0):
        #     raise (RuntimeError(f"No files in: {root}"))

        self.root = root
        self.loader = loader
        self.images = images
        self.transform = transforms.Compose(transforms_)
        self.target_transform = transforms.Compose(target_transform)

    def __getitem__(self, index: int) -> tuple:
        """
        returns image and target as a tuple
        """
        path, target = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(image)

        return image, target

    def __repr__(self) -> str:

        # TODO define representation of class

        return super().__repr__()

    def __len__(self):
        return len(self.images)
