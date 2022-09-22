from datetime import datetime
from os import get_terminal_size, terminal_size

from torchvision.utils import save_image
import torch.cuda as cuda
import os
from PIL import Image


class Visualizer:
    """
    Visialize model's working
    """

    def __init__(self, dataset_imgs_path: str, model_generated_imgs_path: str) -> None:

        self.dataset_imgs_path = dataset_imgs_path
        self.model_genrated_imgs_path = model_generated_imgs_path

        self.dataset_list = self.__get_files_list(self.dataset_imgs_path)
        self.model_generated_list = self.__get_files_list(self.model_genrated_imgs_path)

        self.temp_i1 = "./ISTD_Dataset/test/set_A/90-1.png"
        self.temp_i2 = "./output/B_200/90-1.png"
        # print(self.dataset_list[-1])
        # print(self.model_generated_list[-1])
        # print(self.model_generated_list)
        # i1 = self.__image_loader(self.dataset_imgs_path + "/" + self.dataset_list[0])

        i1 = self.__image_loader((self.dataset_imgs_path + "/" + self.dataset_list[0]))
        i2 = self.__image_loader(
            (self.model_genrated_imgs_path + "/" + self.model_generated_list[0])
        )
        i3 = self.image_concatrate(i1, i2, vertical=False)
        # i3.show()
        # i1.show()
        # i2.show()

    def save_images_list(self, root_path: str, vertical: bool = True) -> None:
        if not os.path.exists(root_path):
            os.makedirs(root_path)
            print("done")
        iteration = 1
        for image_name in self.dataset_list:
            image1 = self.__image_loader(self.dataset_imgs_path + "/" + image_name)
            image2 = self.__image_loader(
                self.model_genrated_imgs_path + "/" + image_name
            )
            combined_image = self.image_concatrate(image1, image2, vertical)
            # combined_image.show()
            combined_image.save(root_path + "/" + image_name)
            print(f"Saved {iteration} of {len(self.dataset_list)}")
            iteration += 1
            # i += 1
            # if i >= 3:
            #     break
        print("DONE")

    def image_concatrate(
        self, image1: Image.Image, image2: Image.Image, vertical: bool = True
    ) -> Image.Image:
        if vertical:
            result_image = Image.new(
                "RGB", (image1.width + image2.width, image1.height)
            )
            result_image.paste(image1, (0, 0))
            result_image.paste(image2, (image1.width, 0))
        else:
            result_image = Image.new(
                "RGB", (image1.width, image1.height + image2.height)
            )
            result_image.paste(image1, (0, 0))
            result_image.paste(image2, (0, image1.height))
        return result_image

    def __image_loader(self, image_path: str, image_scale: float = 1) -> Image.Image:
        """
        loading image by pillow and convert it to RGB
        """
        # https://github.com/python-pillow/Pillow/issues/835

        image = Image.open(image_path)

        width, height = image.size
        # newsize = (int(width*image_scale),int(height*image_scale))
        # image.resize(newsize)

        return image.convert("RGB")

    def __get_files_list(self, folder_path: str) -> list:
        return sorted(os.listdir(folder_path))


# helper functions
def get_current_date_string() -> str:
    """
    getting current date and returning it as a string in \n
    YYYY/MM/DD-HH:MM:SS\n
    format
    """
    date = datetime.now()

    day_string = "{:4d}/{:02d}/{:2d}".format(date.year, date.month, date.day)
    hour_string = "{:02d}:{:02d}:{:02d}".format(date.hour, date.minute, date.second)

    return day_string + "-" + hour_string


def print_memory_status(optional_title_message: str = None) -> None:
    """
    printing status of allocated & cashed memory in cuda
    """

    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET_COLOR = "\033[0;0m"

    terminal_size = get_terminal_size()
    print("\u2500" * terminal_size.columns)

    if optional_title_message:
        print(GREEN + optional_title_message + RESET_COLOR)

    print(
        RED + "CUDA memory allocated" + RESET_COLOR,
        f":\t {cuda.memory_allocated()}",
        sep="",
    )
    print(
        RED + "CUDA memory cashed" + RESET_COLOR,
        f":\t {cuda.memory_reserved()}",
        sep="",
    )
    print("\u2500" * terminal_size.columns)


# ds_imgs = "./data/ISTD_Dataset/test/set_A"
# model_imgs = ".data/results/B_200"

# vis = Visualizer(ds_imgs, model_imgs)
# vis.save_images_list("combined_res_h", vertical=False)
