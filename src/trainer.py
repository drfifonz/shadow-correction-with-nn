import itertools
import models
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import mask_generator, weights_init
from utils.utils import LR_lambda
from utils.utils import QueueMask
from utils.utils import Buffer

# TODO clean up & add comments & rename some variables
# TODO maybe create dedicated discriminator's optimizers
# TODO what should disc_f2s and disc_s2f return
# TODO queue mask and buffer classes


class Trainer:
    """
    Class Trainer creates and instantiates all models, loss criterions, optimizers and
    learning rates. It's functions are responsible for training models, creating optimizers and
    updating learning rates. Additionally it holds methods for saving, loading and resuming training states.
    """

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
            in_channels=opt.in_channels
        )
        self.discriminator_free_to_shadow = models.Discriminator(
            in_channels=opt.out_channels
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

        # self.__critirion_init()

        # self.__optimizers_init()
        # self.__learning_rate_schedulers_init(opt)

    def critirion_init(self) -> tuple:
        """
        initializes loss creterion
        """
        # this criterion uses mean squared error between inputs
        gan_loss_criterion = nn.MSELoss()

        # L1Loss calculates the mean absulute error
        cycle_loss_criterion = nn.L1Loss()
        identity_loss_criterion = nn.L1Loss()

        return gan_loss_criterion, cycle_loss_criterion, identity_loss_criterion

    def learning_rate_schedulers_init(self, opt, current_epoch: int):
        """
        Initializes learning rate schedulers
        """

        # lr schedulers
        lr_scheduler_gen = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_gen,
            lr_lambda=LR_lambda(opt.n_epochs, 0, opt.decay_epoch).step(current_epoch),
        )
        lr_scheduler_disc_s = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_disc_shadower,
            lr_lambda=LR_lambda(opt.n_epochs, 0, opt.decay_epoch).step(current_epoch),
        )
        lr_scheduler_disc_d = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_disc_deshadower,
            lr_lambda=LR_lambda(opt.n_epochs, 0, opt.decay_epoch).step(current_epoch),
        )
        return lr_scheduler_gen, lr_scheduler_disc_s, lr_scheduler_disc_d

    # TODO pamrams types
    def run_one_batch_for_generator(
        self,
        real_shadow: torch.Tensor,
        real_mask: torch.Tensor,
        mask_non_shadow: torch.Tensor,
        mask_queue: QueueMask,
        target_real: torch.Tensor,
        gan_loss_criterion: nn.MSELoss,
        cycle_loss_criterion: nn.L1Loss,
        identity_loss_criterion: nn.L1Loss,
    ):
        self.optimizer_gen.zero_grad()

        same_mask = self.generator_shadow_to_free(real_shadow)
        same_shadow = self.generator_free_to_shadow(real_mask, mask_non_shadow)

        # Identity loss
        identity_loss_mask = identity_loss_criterion(same_mask, real_mask) * 5.0
        identity_loss_shadow = identity_loss_criterion(same_shadow, real_shadow) * 5.0

        # GAN loss
        fake_mask = self.generator_shadow_to_free(real_shadow)
        pred_fake = self.discriminator_free_to_shadow(fake_mask)

        loss_gen_shadow_to_free = gan_loss_criterion(pred_fake, target_real)
        mask_queue.insert(mask_generator(real_shadow, fake_mask))

        fake_shadow = self.generator_free_to_shadow(real_mask, mask_queue.rand_item())
        pred_fake = self.discriminator_shadow_to_free(fake_shadow)
        loss_gen_free_to_shadow = gan_loss_criterion(pred_fake, target_real)

        # Cycle loss
        recovered_shadow = self.generator_free_to_shadow(
            fake_mask, mask_queue.last_item()
        )
        loss_cycle_shadow = cycle_loss_criterion(recovered_shadow, real_shadow) * 10.0

        recovered_mask = self.generator_shadow_to_free(fake_shadow)
        loss_cycle_mask = cycle_loss_criterion(recovered_mask, real_mask) * 10.0

        # Total loss
        gen_loss = (
            identity_loss_shadow
            + identity_loss_mask
            + loss_gen_shadow_to_free
            + loss_gen_free_to_shadow
            + loss_cycle_shadow
            + loss_cycle_mask
        )
        gen_loss.backward()

        self.optimizer_gen.step()

    def run_one_batch_for_discriminator_s2f(
        self,
        real_shadow: torch.Tensor,
        real_mask: torch.Tensor,
        target_real: torch.Tensor,
        target_fake: torch.Tensor,
        fake_shadow_buff: Buffer,
        mask_queue: QueueMask,
        gan_loss_criterion: nn.MSELoss,
    ):
        # zero_grad()
        self.optimizer_disc_deshadower.zero_grad()

        # Real loss
        prediction_real = self.discriminator_shadow_to_free(real_shadow)
        loss_disc_real = gan_loss_criterion(prediction_real, target_real)

        # Fake loss
        fake_shadow = self.generator_free_to_shadow(real_mask, mask_queue.rand_item())
        fake_shadow = fake_shadow_buff.push_and_pop(fake_shadow)
        prediction_fake = self.discriminator_shadow_to_free(fake_shadow.detach())
        loss_disc_fake = gan_loss_criterion(prediction_fake, target_fake)

        # Total loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2.0
        loss_disc.backward()
        self.discriminator_optimizer.step()

    def run_one_batch_for_discriminator_f2s(
        self,
        real_shadow: torch.Tensor,
        real_mask: torch.Tensor,
        target_real: torch.Tensor,
        target_fake: torch.Tensor,
        fake_mask_buff: Buffer,
        mask_queue: QueueMask,
        gan_loss_criterion: nn.MSELoss,
    ):
        # zero_grad()
        self.optimizer_disc_shadower.zero_grad()

        # Real loss
        prediction_real = self.discriminator_free_to_shadow(real_mask)
        loss_disc_real = gan_loss_criterion(prediction_real, target_real)

        # Fake loss
        fake_mask = self.generator_shadow_to_free(real_shadow, mask_queue.rand_item())
        fake_mask = fake_mask_buff.push_and_pop(fake_mask)
        prediction_fake = self.discriminator_free_to_shadow(fake_mask.detach())
        loss_disc_fake = gan_loss_criterion(prediction_fake, target_fake)

        # Total loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2.0
        loss_disc.backward()
        self.discriminator_optimizer.step()

    # TODO description
    def discriminator_optimizer(
        self,
        model: nn.Module,
    ):
        return torch.optim.Adam(model.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))

    # TODO description
    def generator_optimizer(
        self,
        model_deshadower: nn.Module,
        model_shadower: nn.Module,
    ):
        combine_parameters = itertools.chain(
            model_deshadower.parameters(), model_shadower.parameters()
        )
        return torch.optim.Adam(combine_parameters, lr=self.opt.lr, betas=(0.5, 0.999))

    def allocate_memory(opt):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        input_shadow = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)
        input_mask = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)
        target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)
        mask_non_shadow = Variable(
            Tensor(opt.batch_size, 1, opt.size, opt.size).fill_(-1.0),
            requires_grad=False,
        )
        return [input_shadow, input_mask, target_real, target_fake, mask_non_shadow]

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
