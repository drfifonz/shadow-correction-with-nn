import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from dotenv import load_dotenv

import detection.transforms as T
import detection.utils as utils
from detection.engine import evaluate, train_one_epoch
from pretrained_seg_model import pretrained_seg_model
from ShadowDataset import ShadowDataset
from ShadowModel import ShadowModel

load_dotenv()  # loading env variables from .env file
TEST_ROOT_PATH = "data/ISTD_Dataset/test"
TRAIN_ROOT_PATH = "data/ISTD_Dataset/train"


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # emptying torch cache to free GPU's memory
    torch.cuda.empty_cache()

    # selecting GPU as the device for calculations
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # dataset has two classes only - background and shadow
    num_classes = 2

    # use dataset and defined transformations
    dataset = ShadowDataset(TRAIN_ROOT_PATH, get_transform(train=True))
    dataset_test = ShadowDataset(TEST_ROOT_PATH, get_transform(train=True))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    indices_test = torch.randperm(len(dataset_test)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=utils.collate_fn,
    )

    # get the model using helper function
    model = pretrained_seg_model(num_classes)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 3  # just for testing

    for epoch in range(num_epochs):

        # train for one epoch, printing every 20 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        res = evaluate(model, data_loader_test, device=device)
        if epoch == num_epochs - 1:
            print(res)
    print("It's done!")


if __name__ == "__main__":
    main()
