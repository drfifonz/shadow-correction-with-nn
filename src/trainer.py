import torch

import models


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

        if torch.cuda.is_available():
            self.generator_free_to_shadow.cuda()
            self.generator_shadow_to_free.cuda()
            self.discriminator_free_to_shadow.cuda()
            self.discriminator_shadow_to_free.cuda()


    def run_one_batch_for_generator(self, data):
        pass
        #zero_grad()
        # Identity loss
        # GAN loss
        # Cycle loss
        #Total loss

    def run_one_batch_for_discriminator(self,data):
        #zero_grad()
        # Real loss
        # Fake loss
        # Total loss
        pass

    def update_lr():
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

    