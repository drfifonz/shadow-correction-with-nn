import os
from unittest import result

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np

from models import Generator_F2S, Generator_S2F
from utils.utils import mask_generator, QueueMask


def test(opt):

    dataset_shadow_path = "./data/ISTD_Dataset/test/set_A"
    dataset_shadow_free_path = "./data/ISTD_Dataset/test/set_C"

    result_path = "./data/results1"
    im_sufix = ".png"

    generator_deshadower = "./data/results/generator_shadow_to_free_200.pth"
    generator_shadower = "./data/results/generator_free_to_shadow_200.pth"

    # raise "OK"
    if torch.cuda.is_available():
        opt.cuda = True
        device = torch.device("cuda:0")

    print(opt)
    # print("hi\n\n\n")

    ###### Definition of variables ######
    # Networks
    # Deshadower = Generator_S2F(opt.in_channels, opt.out_channels)
    Deshadower = Generator_S2F(opt.in_channels, opt.out_channels)
    Shadower = Generator_F2S(opt.out_channels, opt.in_channels)

    if opt.cuda:
        Deshadower.to(device)
        Shadower.to(device)

    # Load state dicts
    Deshadower.load_state_dict(torch.load(generator_deshadower))
    Shadower.load_state_dict(torch.load(generator_shadower))

    # Set model's test mode
    Deshadower.eval()
    Shadower.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.in_channels, opt.size, opt.size)
    input_B = Tensor(opt.batch_size, opt.out_channels, opt.size, opt.size)

    # Dataset loader
    img_transform = transforms.Compose(
        [
            transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    to_pil = transforms.ToPILImage()

    ###### Testing######

    # Create output dirs
    if not os.path.exists(f"{result_path}/A"):
        os.makedirs(f"./{result_path}/A")
    if not os.path.exists(f"{result_path}/B"):
        os.makedirs(f"{result_path}/B")
    if not os.path.exists(f"{result_path}/mask"):
        os.makedirs(f"{result_path}/mask")

    # Deshadower
    images_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(dataset_shadow_path)
        if f.endswith(im_sufix)
    ]

    mask_queue = QueueMask(len(images_list))

    for index, img_name in enumerate(images_list):

        img = Image.open(
            os.path.join(dataset_shadow_path, img_name + im_sufix)
        ).convert("RGB")
        width, height = img.size

        image_variable = (img_transform(img).unsqueeze(0)).to(device)

        temp_B = Deshadower(image_variable)

        fake_B = 0.5 * (temp_B.data + 1.0)
        mask_queue.insert(mask_generator(image_variable, temp_B))
        fake_B = np.array(
            transforms.Resize((height, width))(to_pil(fake_B.data.squeeze(0).cpu()))
        )

        Image.fromarray(fake_B).save(result_path + f"/B/{img_name + im_sufix}")

        print(f"Generated images {(index + 1):03d} of {len(images_list):03d}")

    # Shadower
    images_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(dataset_shadow_free_path)
        if f.endswith(im_sufix)
    ]

    for index, img_name in enumerate(images_list):

        img = Image.open(
            os.path.join(dataset_shadow_free_path, img_name + im_sufix)
        ).convert("RGB")
        width, height = img.size

        mask = mask_queue.rand_item()
        mask_cpu = np.array(
            transforms.Resize((height, width))(
                to_pil(((mask.data + 1) * 0.5).squeeze(0).cpu())
            )
        )

        temp_A = Shadower(image_variable, mask)

        fake_A = 0.5 * (temp_A.data + 1.0)

        fake_A = np.array(
            transforms.Resize((height, width))(to_pil(fake_A.data.squeeze(0).cpu()))
        )

        # Save image files

        Image.fromarray(fake_A).save(result_path + f"/A/{img_name + im_sufix}")

        Image.fromarray(mask_cpu).save(result_path + f"/mask/{img_name + im_sufix}")

        print(f"Generated images {(index + 1):03d} of {len(images_list):03d}")
