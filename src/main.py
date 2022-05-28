import os
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from ShadowModel import ShadowModel
from ShadowDataset import ShadowDataset
from detection.engine import train_one_epoch, evaluate

# import torchvision.transforms as T

import detection.transforms as T
from pretrained_seg_model import pretrained_seg_model
import detection.utils as utils

TEST_ROOT_PATH = "data/ISTD_Dataset/test"
TRAIN_ROOT_PATH = "data/ISTD_Dataset/train"


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.Resize)
    return T.Compose(transforms)


def main():
    device = torch.device("cpu")

    # dataset has two classes only - background and shadow
    num_classes = 2
    # use dataset and defined transformations
    dataset = ShadowDataset(TRAIN_ROOT_PATH, get_transform(train=True))
    dataset_test = ShadowDataset(TEST_ROOT_PATH, get_transform(train=True))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    model = pretrained_seg_model(num_classes)

    model.to(device="cpu")

    #    construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    main()
