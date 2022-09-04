
import os
import random

import torch
import torchvision.transforms as transforms

from PIL import Image


class ISTD_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_list:list=None, unaligned:bool=False, mode:str="train") -> None:
        self.transform = transforms.Compose(transforms_list)
        self.unaligned = unaligned
        self.root_shadow_imgs = root + "/" + mode + "/set_A"
        self.root_shadow_free_imgs = root + "/" + mode + "/set_C"

        self.shadow_files = sorted(os.listdir(self.root_shadow_imgs))
        self.shadow_free_files = sorted(os.listdir(self.root_shadow_free_imgs))



    def __getitem__(self, index):


        item_shadow = self.transform(
            Image.open(
                self.root_shadow_imgs
                + "/"
                + self.shadow_files[index % len(self.shadow_files)]
            )
        )

        if self.unaligned:
            item_shadow_free = self.transform(
                self.__image_loader(
                    self.shadow_free_files[
                        random.randint(0, len(self.shadow_free_files) - 1)
                    ]
                )
            )
        else:
            item_shadow_free = self.transform(

                self.__image_loader(
                    self.root_shadow_free_imgs
                    + "/"
                    + self.shadow_free_files[index % len(self.shadow_free_files)]

                )

            )

        return {"Shadow": item_shadow, "Shadow-free": item_shadow_free}

    def __len__(self):
                
        return max(len(self.shadow_files), len(self.shadow_free_files))

    def __image_loader(self,image_path: str)-> Image.Image:
        """
        loading image by pillow and convert it to RGB
        """
        # https://github.com/python-pillow/Pillow/issues/835
        return Image.open(image_path).convert("RGB")


    