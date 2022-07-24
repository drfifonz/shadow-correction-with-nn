import itertools
import torch
import torch.nn as nn

import models
from utils.utils import weights_init


class Trainer:
    def __init__(self, opt) -> None:
        self.opt = opt
        # networks
        self.generator_shadow_to_free = models.Deshadower(
            in_channels=opt.in_channels, out_channels=opt.out_channels
        )
        self.generator_free_to_shadow = models.Shadower(
            in_channels=opt.in_channels, out_channels=opt.out_channels
        )
        self.discriminator_shadow_to_free = models.Discriminator(
            input_nc=opt.in_channels
        )
        self.discriminator_free_to_shadow = models.Discriminator(
            input_nc=opt.in_channels
        )

        # sending models to gpu
        if torch.cuda.is_available():
            self.generator_free_to_shadow.cuda()
            self.generator_shadow_to_free.cuda()
            self.discriminator_free_to_shadow.cuda()
            self.discriminator_shadow_to_free.cuda()

        # applying weights init
        self.generator_free_to_shadow.apply(weights_init)
        self.generator_shadow_to_free.apply(weights_init)
        self.discriminator_free_to_shadow.apply(weights_init)
        self.discriminator_shadow_to_free.apply(weights_init)

        # TODO losses criterion

        # optimizer init
        self.optimizer_gen = self.generator_optimizer(
            self.generator_shadow_to_free, self.generator_free_to_shadow
        )
        self.optimizer_disc_deshadower = self.discriminator_optimizer(
            self.discriminator_shadow_to_free
        )
        self.optimizer_disc_shadower = self.discriminator_optimizer(
            self.discriminator_free_to_shadow
        )

    def memory_allocation(self):
        # allocate memory
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        # TODO to be finished
        pass

    def run_one_batch_for_generator(self, data):
        pass
        # zero_grad()
        self.optimizer_gen.zero_grad()
        # Identity loss
        # GAN loss
        # Cycle loss
        # Total loss
        self.optimizer_gen.step()

    def run_one_batch_for_discriminator(self, data):
        # zero_grad()
        # Real loss
        # Fake loss
        # Total loss
        pass

    def discriminator_optimizer(
        self,
        model: nn.Module,
    ):
        return torch.optim.Adam(model.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))

    def generator_optimizer(
        self,
        model_deshadower: nn.Module,
        model_shadower: nn.Module,
    ):
        combine_parameters = itertools.chain(
            model_deshadower.parameters(), model_shadower.parameters()
        )
        return torch.optim.Adam(combine_parameters, lr=self.opt.lr, betas=(0.5, 0.999))

    def update_lr_per_epoch():
        pass

    def update_lr_per_batch():
        pass

    def save_training_state(self, training_state_path, epoch_n, *networks):
        """
        saving state of networks after epoch training
        https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """

        pass

    def load_training_state(self, training_state_path, *networks):
        """
        load state from certain epoch
        """
        pass

    def resume_training_state(self, anything):
        """
        resume state from certain epoch
        """
        pass
