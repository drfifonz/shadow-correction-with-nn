import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from trainer import Trainer
from utils.visualizer import Visualizer
from utils.utils import QueueMask, Buffer


def train(opt):
    """
    training model
    """

    if opt.resume:
        epoch_start = 3
    else:
        epoch_start = 0

    dataloader = DataLoader()

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
        for i, data in enumerate(dataloader, start=epoch_start):

            # set model input
            real_shadow = Variable(input_shadow.copy_(data["A"]))
            real_mask = Variable(input_mask.copy_(data["B"]))

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
