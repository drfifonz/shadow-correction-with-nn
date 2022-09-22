import os
import sys
import matplotlib.pyplot as plt
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

torch.cuda.empty_cache()

load_dotenv()

# istd = "./data/ISTD_Dataset"
ISTD_PATH = os.environ.get("ISTD_DATASET_ROOT_PATH")


def train(opt):
    """
    training model
    """

    trainer = Trainer(opt)
    visualizer = Visualizer(opt)

    (
        lr_scheduler_gen,
        lr_scheduler_disc_s,
        lr_scheduler_disc_d,
    ) = trainer.learning_rate_schedulers_init(opt, current_epoch=0)

    if opt.resume:

        trainer.resume_training_state(
            "./saved_training_state",
            [
                trainer.generator_free_to_shadow,
                trainer.generator_shadow_to_free,
                trainer.discriminator_free_to_shadow,
                trainer.discriminator_shadow_to_free,
            ],
            [
                trainer.optimizer_gen,
                trainer.optimizer_disc_deshadower,
                trainer.optimizer_disc_shadower,
            ],
            [lr_scheduler_gen, lr_scheduler_disc_s, lr_scheduler_disc_d],
        )
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
        # ISTD_Dataset(root=istd, transforms_list=transformation_list)
        ISTD_Dataset(root=ISTD_PATH, transforms_list=transformation_list)
    )

    # memory allocation
    (
        input_shadow,
        input_mask,
        target_real,
        target_fake,
        mask_non_shadow,
    ) = Trainer.allocate_memory(opt)

    mask_queue = QueueMask(len(dataloader) // 4)
    # print("len: ", len(dataloader) // 4)
    fake_shadow_buff = Buffer()
    fake_mask_buff = Buffer()

    (
        gan_loss_criterion,
        cycle_loss_criterion,
        identity_loss_criterion,
    ) = Trainer.critirion_init()

    # iteration counter
    current_it = 0
    plt.ioff()
    to_pil = transforms.ToPILImage()

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

            (
                gen_loss,
                identity_loss_mask,
                identity_loss_shadow,
                loss_gen_shadow_to_free,
                loss_gen_free_to_shadow,
                loss_cycle_mask,
                loss_cycle_shadow,
                fake_shadow,
                fake_mask,
                gen_losses_temp,
            ) = trainer.run_one_batch_for_generator(
                real_shadow,
                real_mask,
                mask_non_shadow,
                mask_queue,
                target_real,
                gen_losses_temp,
                gan_loss_criterion,
                cycle_loss_criterion,
                identity_loss_criterion,
            )
            total_loss_disc_s2f = 0
            (
                loss_disc_s2f,
                disc_s2f_losses_temp,
            ) = trainer.run_one_batch_for_discriminator_s2f(
                real_shadow,
                real_mask,
                target_real,
                target_fake,
                fake_shadow_buff,
                mask_queue,
                gan_loss_criterion,
                disc_s2f_losses_temp,
                fake_shadow,
            )
            total_loss_disc_s2f += loss_disc_s2f.detach()
            # del loss_disc_s2f

            total_loss_disc_f2s = 0
            (
                loss_disc_f2s,
                disc_f2s_losses_temp,
            ) = trainer.run_one_batch_for_discriminator_f2s(
                real_shadow,
                real_mask,
                target_real,
                target_fake,
                fake_mask_buff,
                mask_queue,
                gan_loss_criterion,
                disc_f2s_losses_temp,
                fake_mask,
            )
            total_loss_disc_f2s += loss_disc_f2s.detach()
            # del loss_disc_f2s

            current_it += 1
            print(f"current_it: \t {current_it}")
            if (i + 1) % opt.iteration_loss == 0:
                print(
                    f"[Iteration: {current_it:d}], [gen_loss: {gen_loss:.5f}], [loss_identity_gen: {(identity_loss_shadow + identity_loss_mask):.5f}], [loss_gen_s2f_and_f2s: {(loss_gen_free_to_shadow+loss_gen_shadow_to_free):.5f}],"
                    f"[Cycle_loss: {(loss_cycle_shadow + loss_cycle_mask):.5f}], [loss_disc: {(total_loss_disc_f2s + total_loss_disc_s2f):.5f}]"
                )

                gen_losses.append(gen_losses_temp / opt.iteration_loss)
                disc_s2f_losses.append(disc_s2f_losses_temp / opt.iteration_loss)
                disc_f2s_losses.append(disc_f2s_losses_temp / opt.iteration_loss)

                # reseting temporary losses
                gen_losses_temp = 0
                disc_s2f_losses_temp = 0
                disc_f2s_losses_temp = 0

                # print("Avarage log: \n")
                print(
                    f"[Last {opt.iteration_loss} iterations], [gen loss: {(gen_losses[len(gen_losses)-1]):.5f}], [disc s2f loss: {disc_s2f_losses[len(disc_s2f_losses)-1]}], disc f2s loss: {(disc_f2s_losses[len(disc_f2s_losses)-1]):.5f}"
                )

                img_fake_shadow = 0.5 * (fake_shadow.detach().data + 1.0)
                img_fake_shadow = to_pil(img_fake_shadow.data.squeeze(0).cpu())
                img_fake_shadow.save("output/fake_A.png")

                img_fake_mask = 0.5 * (fake_mask.detach().data + 1.0)
                img_fake_mask = to_pil(img_fake_mask.data.squeeze(0).cpu())
                img_fake_mask.save("output/fake_B.png")

            # update learning rates

            lr_scheduler_gen.step()
            lr_scheduler_disc_s.step()
            lr_scheduler_disc_d.step()

            # torch.save(
            #     trainer.generator_shadow_to_free.state_dict(),
            #     "./data/results1/netG_A2B.pth",
            # )
            # torch.save(
            #     trainer.generator_free_to_shadow.state_dict(),
            #     "./data/results1/netG_B2A.pth",
            # )
            # torch.save(
            #     trainer.discriminator_shadow_to_free.state_dict(),
            #     "./data/results1/discriminator_shadow_to_free.pth",
            # )
            # torch.save(
            #     trainer.discriminator_free_to_shadow.state_dict(),
            #     "./data/results1/discriminator_free_to_shadow.pth",
            # )

            # torch.save(
            #     trainer.optimizer_gen.state_dict(), "./data/results1/optimizer_gen.pth"
            # )
            # torch.save(
            #     trainer.optimizer_disc_deshadower.state_dict(),
            #     "./data/results1/optimizer_disc_deshadower.pth",
            # )
            # torch.save(
            #     trainer.optimizer_disc_shadower.state_dict(),
            #     "./data/results1/optimizer_disc_shadower.pth",
            # )

            # torch.save(
            #     lr_scheduler_gen.state_dict(), "./data/results1/lr_scheduler_gen.pth"
            # )
            # torch.save(
            #     lr_scheduler_disc_s.state_dict(), "./data/results1/ lr_scheduler_disc_s.pth"
            # )
            # torch.save(
            #     lr_scheduler_disc_d.state_dict(), "./data/results1/lr_scheduler_disc_d.pth"
            # )

            if (epoch + 1) % opt.snapshot_epochs == 0:
                torch.save(
                    trainer.generator_shadow_to_free.state_dict(),
                    ("./data/results1/generator_shadow_to_free_%d.pth" % (epoch + 1)),
                )
                torch.save(
                    trainer.generator_free_to_shadow.state_dict(),
                    ("./data/results1/generator_free_to_shadow_%d.pth" % (epoch + 1)),
                )
                torch.save(
                    trainer.discriminator_shadow_to_free.state_dict(),
                    (
                        "./data/results1/discriminator_shadow_to_free_%d.pth"
                        % (epoch + 1)
                    ),
                )
                torch.save(
                    trainer.discriminator_free_to_shadow.state_dict(),
                    (
                        "./data/results1/discriminator_free_to_shadow_%d.pth"
                        % (epoch + 1)
                    ),
                )

    print(f"Epoch: {epoch+1} finished")


print("Finished training loop.")
