import os
import sys

import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloaders.ISTD_dataset import ISTD_Dataset
from trainer import Trainer
from utils.utils import Buffer, QueueMask
from utils.visualizer import Visualizer


load_dotenv()

ISTD_PATH = os.environ.get("ISTD_DATASET_ROOT_PATH")


def train(opt):
    """
    training model
    """

    if opt.resume:
        # temporary solution
        epoch_start = 3
    else:
        epoch_start = 0

    transformation_list = [
        # transforms.Resize((opt.size, opt.size), Image.BICUBIC),
        transforms.Resize(int(400 * 1.12), Image.BICUBIC),
        transforms.RandomCrop(400),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ISTD_Dataset(root=ISTD_PATH, transforms_list=transformation_list)
    )

    trainer = Trainer(opt)
    visualizer = Visualizer(opt)

    # memory allocation
    (
        input_shadow,
        input_mask,
        target_real,
        target_fake,
        mask_non_shadow,
    ) = Trainer.allocate_memory(opt)

    # mask queue
    mask_queue = QueueMask(dataloader.__len__() / 4)
    fake_shadow_buff = Buffer()
    fake_mask_buff = Buffer()

    # TRAINING
    print("Starting training loop...")
    for epoch in range(epoch_num := opt.epochs):
        for i, data in enumerate(dataloader):

            # set model input
            real_shadow = Variable(input_shadow.copy_(data["Shadow"]))
            real_mask = Variable(input_mask.copy_(data["Shadow-free"]))

            # training part
            trainer.run_one_batch_for_generator(
                real_shadow, real_mask, mask_non_shadow, mask_queue, target_real
            )
            trainer.run_one_batch_for_discriminator_s2f(
                real_shadow,
                real_mask,
                target_real,
                target_fake,
                fake_shadow_buff,
                mask_queue,
            )
            trainer.run_one_batch_for_discriminator_f2s(
                real_shadow,
                real_mask,
                target_real,
                target_fake,
                fake_mask_buff,
                mask_queue,
            )
            # discriminator can be used less time than generator

            # visualization part (plots.etc)
            # terminal plotting (probably need new class)

        # update lr
        # saving epoch state
