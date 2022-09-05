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

    # iteration counter
    current_it = 0

    # temporary losses
    gen_losses_temp = 0
    disc_s2f_losses_temp = 0
    disc_f2s_losses_temp = 0

    # avarage losses
    gen_losses = []
    disc_s2f_losses = []
    disc_f2s_losses = []

    # TRAINING
    print("Starting training loop...")
    for epoch in range(epoch_num := opt.epochs):
        for i, data in enumerate(dataloader):

            # set model input
            real_shadow = Variable(input_shadow.copy_(data["Shadow"]))
            real_mask = Variable(input_mask.copy_(data["Shadow-free"]))

            # training part
            # TODO changed for log
            (
                gen_loss,
                identity_loss_mask,
                identity_loss_shadow,
                loss_gen_shadow_to_free,
                loss_gen_free_to_shadow,
                loss_cycle_mask,
                loss_cycle_shadow,
            ) = trainer.run_one_batch_for_generator(
                real_shadow,
                real_mask,
                mask_non_shadow,
                mask_queue,
                target_real,
                gen_losses_temp,
            )

            loss_disc_s2f = trainer.run_one_batch_for_discriminator_s2f(
                real_shadow,
                real_mask,
                target_real,
                target_fake,
                fake_shadow_buff,
                mask_queue,
                disc_s2f_losses_temp,
            )

            loss_disc_f2s = trainer.run_one_batch_for_discriminator_f2s(
                real_shadow,
                real_mask,
                target_real,
                target_fake,
                fake_mask_buff,
                mask_queue,
                disc_f2s_losses_temp,
            )
            # discriminator can be used less time than generator

            # visualization part (plots.etc)

            # terminal plotting (probably need new class)
            current_it += 1
            if (i + 1) % opt.iteration_loss == 0:
                print(
                    f"Iteration: {current_it}, gen loss: {gen_loss}, loss identity gen: {identity_loss_shadow + identity_loss_mask}, loss gen s2f and f2s: {loss_gen_free_to_shadow+loss_gen_shadow_to_free},"
                    f"Cycle loss: {loss_cycle_shadow + loss_cycle_mask}, loss disc: {loss_disc_f2s + loss_disc_s2f}"
                )

                gen_losses.append(gen_losses_temp / opt.iteration_loss)
                disc_s2f_losses.append(disc_s2f_losses_temp / opt.iteration_loss)
                disc_f2s_losses.append(disc_f2s_losses_temp / opt.iteration_loss)

                # reseting temporary losses
                gen_losses_temp = 0
                disc_s2f_losses_temp = 0
                disc_f2s_losses_temp = 0

                print("Avarage log: \n")
                print(
                    f"Last {opt.iteration_loss} iterations: \n, gen loss: {gen_losses[gen_losses.__len__()-1]}, disc s2f loss: {disc_s2f_losses[disc_s2f_losses.__len__()-1]}, disc f2s loss: {disc_f2s_losses[disc_f2s_losses.__len__()-1]}"
                )

                # saving images

        # update lr
        # saving epoch state
