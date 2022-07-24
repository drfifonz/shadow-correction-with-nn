import torch
from torch.utils.data import DataLoader

from trainer import Trainer
from utils.visualizer import Visualizer


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

    # TRAINING
    print("Starting training loop...")
    for epoch in range(epoch_num := opt.epochs):
        for i, data in enumerate(dataloader, start=epoch_start):

            # training part
            trainer.run_one_batch_for_generator(data=data)
            trainer.run_one_batch_for_discriminator(data=data)
            # discriminator can be used less time than generator

            # visualization part (plots.etc)
            # terminal plotting (probably need new class)

        # update lr
        # saving epoch state
