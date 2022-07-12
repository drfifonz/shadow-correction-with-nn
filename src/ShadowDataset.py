import os
import random
from re import I

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

SHADOW_DATASET_PATH = "set_A"
MASK_DATASET_PATH = "set_B"
CLEAR_DATASET_PATH = "set_C"
# TEST_SHADOW_DATASET_PATH = "test/test_A"
# TEST_MASK_DATASET_PATH = "test/test_B"
# TEST_CLEAR_DATASET_PATH = "test/test_C"
def rename_paths(dataset_root_path: str) -> str:
    """
    checking and renaming folder's path to standarized vesion
    """
    pass


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


class ShadowDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms) -> None:
        # dtype = torch.float32
        self.root = root
        self.transforms = transforms

        # loading images
        self.shadow_images = list(
            sorted(os.listdir(os.path.join(root, SHADOW_DATASET_PATH)))
        )
        self.shadow_masks = list(
            sorted(os.listdir(os.path.join(root, MASK_DATASET_PATH)))
        )

        dataset_perc = 0.2  # part of dataset
        self.shadow_images = self.shadow_images[
            : int(len(self.shadow_images) * dataset_perc)
        ]
        self.shadow_masks = self.shadow_masks[
            : int(len(self.shadow_masks) * dataset_perc)
        ]

    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar

    def transform(self, image, mask):

        # TODO
        # needs to be adjusted to dataset layout

        # Resize
        resize = transforms.Resize(size=(128, 128))
        image = resize(image)
        # print(type(image))

        mask = resize(mask)

        # Random crop
        # TODO change output_size to sth more suitable
        # smaller values
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(50, 50))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        # loading images and masks
        shadow_image_path = os.path.join(
            self.root, SHADOW_DATASET_PATH, self.shadow_images[index]
        )
        shadow_mask_path = os.path.join(
            self.root, MASK_DATASET_PATH, self.shadow_masks[index]
        )
        # converting images to RGB
        img = Image.open(shadow_image_path).convert("RGB")

        # converting masks to np.array
        mask = Image.open(shadow_mask_path)  # .convert("RGB")
        mask = np.array(mask)

        shadow_ids = np.unique(mask)

        # 1st id is background, so it is removed
        shadow_ids = shadow_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == shadow_ids[:, None, None]

        # get bounding box coordinates for each mask

        # img, mask = self.transform(img, mask)

        num_shadows = len(shadow_ids)
        boxes = []
        for i in range(num_shadows):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin != xmax and ymin != ymax:
                boxes.append([xmin, ymin, xmax, ymax])

        # converting everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_shadows,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_shadows,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.shadow_images)


class Dataset(data.Dataset):
    def __init__(self, root, loader, transforms_=None, target_transform=None):

        images = create_dataset(root)
        if len(images == 0):
            raise (RuntimeError(f"No files in: {root}"))

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
