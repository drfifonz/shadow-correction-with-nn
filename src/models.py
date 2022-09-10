import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any


class Deshadower(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        in_channels: int = 3,
        res_blocks: int = 9,
        downsampling_iterations: int = 2,
        upsampling_iterations: int = 2,
    ):
        super(Deshadower, self).__init__()

        # Initial conv layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=(out_features := 64),
                kernel_size=7,
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # downsampling
        # in_channels = 64
        # out_channels = in_channels * 2

        in_features = out_features
        out_features = in_features * 2

        # print(f"in_features: {in_features}, out_features: {out_features}\tOK\n ")

        downsampling_block = (
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

        for num in range(downsampling_iterations):
            # in_channels = out_channels
            # out_channels = in_channels * 2
            self.model, *_ = map(
                self.model.append, self.__downsampling_block(in_features, out_features)
            )
            in_features = out_features
            out_features = in_features * 2
            print(
                # f"in_features: {in_features}, out_features: {out_features}\t after downsampling num: {num+1} \n "
            )

        print(
            # f"in_features: {in_features}, out_features: {out_features}\t after downsampling\n "
        )

        # residual blocks
        residual_block = (
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
        )

        if res_blocks <= 0:
            raise Exception(
                f"res_blocks number should be positive number (it's: {res_blocks})"
            )

        # in_features = out_features
        for num in range(res_blocks):
            self.model, *_ = map(
                self.model.append, self.__residual_block(in_features, out_features)
            )
            print(
                # f"in_features: {in_features}, out_features: {out_features}\tafter residual num: {num+1}\n "
            )

        # print(
        #     f"in_features: {in_features}, out_features: {out_features}\tafter residual\n "
        # )

        # upsampling
        # out_channels = in_channels // 2
        # in_features = out_features

        out_features = in_features // 2

        print(
            # f"in_features: {in_features}, out_features: {out_features}\tb4 upsampling\n "
        )
        # print("upsampling block")
        upsampling_block = (
            nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

        for num in range(upsampling_iterations):
            self.model, *_ = map(
                self.model.append, self.__upsampling_block(in_features, out_features)
            )
            in_features = out_features
            out_features = in_features // 2
            # in_channels = out_channels
            # out_channels = in_channels // 2
            print(
                # f"in_features: {in_features}, out_features: {out_features}\tafter upsampling num: {num+1}\n "
            )
        # print("------------------------------------")
        # print(self.model)
        # print("------------------------------------")
        # raise
        # output layer
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(64, out_channels, 7))

        # print("------------------------------------")
        print(self.model)
        # print("DESHADOWER")
        print("------------------------------------")
        # raise

    def forward(self, x: torch.Tensor):
        # print(x.shape)

        # could be something wrong with forward function

        return (self.model(x) + x).tanh

    def __downsampling_block(self, in_features: int, out_features: int) -> tuple:
        ds_block = (
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        return ds_block

    def __residual_block(self, in_features: int, out_features: int) -> tuple:
        residual_block = (
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
        )
        return residual_block

    def __upsampling_block(self, in_features: int, out_features: int) -> tuple:
        upsampling_block = (
            nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        return upsampling_block


class Shadower(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        in_channels: int = 3,
        res_blocks=9,
        downsampling_iterations: int = 2,
        upsampling_iterations: int = 2,
    ):
        super(Shadower, self).__init__()

        in_features = in_channels
        out_features = 64
        print(
            # f"in_features: {in_features}, out_features: {out_features}\tb4 initial conv\n "
        )
        # initial conv layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            # Additional channel is added to in_channels
            # because of the presence of the mask
            nn.Conv2d(in_features := in_features + 1, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        print(
            f"in_features: {in_features}, out_features: {out_features}\tafter initial conv\n "
        )

        # downsampling
        # out_channels = in_channels * 2
        in_features = out_features
        out_features = in_features * 2
        print(
            f"in_features: {in_features}, out_features: {out_features}\tb4 downsampling\n "
        )

        downsampling_block = (
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        for num in range(downsampling_iterations):
            self.model, *_ = map(
                self.model.append, self.__downsampling_block(in_features, out_features)
            )
            in_features = out_features
            out_features = in_features * 2
            print(
                f"in_features: {in_features}, out_features: {out_features}\tafter downsampling num: {num+1}\n "
            )
            # in_channels = out_channels
            # out_channels = in_channels * 2

        # residual blocks
        residual_block = (
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )
        if res_blocks <= 0:
            raise Exception(
                f"res_blocks number should be positive number (it's: {res_blocks})"
            )

        # in_features = out_features
        for _ in range(res_blocks):
            self.model, *_ = map(
                self.model.append, self.__residual_block(in_features, out_features)
            )

        print(
            f"in_features: {in_features}, out_features: {out_features}\tafter residual\n "
        )

        # upsampling
        out_features = in_features // 2
        upsampling_block = (
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for _ in range(upsampling_iterations):
            self.model, *_ = map(
                self.model.append, self.__upsampling_block(in_features, out_features)
            )
            in_features = out_features
            out_features = in_features // 2

            print(
                f"in_features: {in_features}, out_features: {out_features}\tafter upsampling num: {num+1}\n "
            )

        # in_channels = out_channels
        # out_channels = in_channels // 2

        # output layer
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(64, out_channels, 7))

        print("------------------------------------")
        print(self.model)
        print("SHADOWER")

        print("------------------------------------")
        # raise

    def forward(self, x: Any, mask):
        """
        Forward method \n
        returns \n
        (self.model(torch.cat((x, mask), 1)) + x).tanh()"""

        print(x.size(), mask.size(), sep="\n")
        return (self.model(torch.cat((x, mask), 1)) + x).tanh()

    def __residual_block(self, in_features: int, out_features: int) -> tuple:
        residual_block = (
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )
        return residual_block

    def __upsampling_block(self, in_features: int, out_features: int) -> tuple:
        upsampling_block = (
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        return upsampling_block

    def __downsampling_block(self, in_features: int, out_features: int) -> tuple:
        downsampling_block = (
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        return downsampling_block


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers_number: int = 3,
    ) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels := 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        downsampling_block = (
            nn.Conv2d(
                in_channels := out_channels,
                out_channels := out_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_channels),  # try nn.BatchNorm2d()
            nn.LeakyReLU(0.2, inplace=True),
        )
        if layers_number <= 0:
            raise Exception("layers_number should be greater than 0")

        for _ in range(layers_number):
            self.model, *_ = map(self.model.append, downsampling_block)

        # classification layer
        self.model.append(
            nn.Conv2d(
                in_channels := out_channels,
                out_channels := 1,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

    def forward(self, x: Any):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(
            x.size()[0], -1
        )  # consider other forward function
