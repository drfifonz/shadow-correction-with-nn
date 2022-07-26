import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
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

    # allocate memory
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    input_shadow = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)
    input_mask = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)
    mask_non_shadow = Variable(
        Tensor(opt.batch_size, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False
    )

    # TRAINING
    print("Starting training loop...")
    for epoch in range(epoch_num := opt.epochs):
        for i, data in enumerate(dataloader, start=epoch_start):

            # set model input
            real_shadow = Variable(input_shadow.copy_(data["A"]))
            real_mask = Variable(input_mask.copy_(data["B"]))

            # training part
            trainer.run_one_batch_for_generator(data=data)
            trainer.run_one_batch_for_discriminator(data=data)
            # discriminator can be used less time than generator

            # visualization part (plots.etc)
            # terminal plotting (probably need new class)

        # update lr
        # saving epoch state
