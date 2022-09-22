import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator_S2F(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual_blocks=9):
        super(Generator_S2F, self).__init__()
        input_nc = in_channels
        output_nc = out_channels

        # Initial layer
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (self.model(x) + x).tanh()


class Generator_F2S(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual_blocks=9):
        super(Generator_F2S, self).__init__()
        input_nc = in_channels
        output_nc = out_channels
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc + 1, 64, 7),  # extra input for mask channel
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]

        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        with torch.no_grad():
            output = (self.model(torch.cat((x, mask), 1)) + x).tanh()

        return output


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers_number: int = 3,
    ) -> None:
        super(Discriminator, self).__init__()

        # print("_______________________________")
        # print("DISCRIMINATOR")

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_features := 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # temp
        out_channels = 64

        in_features = out_features
        out_features = in_features * 2
        # print(f"in_features:\t{in_features}\tout_features\t{out_features}\tafter init")
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

        for num in range(layers_number):
            # print(f"ALOHA {num} of {layers_number}")
            self.model, *_ = map(
                self.model.append, self.__downsampling_block(in_features, out_features)
            )
            # print(
            #     f"in_features:\t{in_features}\tout_features\t{out_features}\tlayer num:\t{num+1}"
            # )
            in_features = out_features
            out_features *= 2
        # classification layer
        self.model.append(
            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        # print(f"in_features:\t{512}\tout_features\t{1}\tlast layer ")
        # print("_______________________________")

    def forward(self, x: Any):
        # with torch.no_grad():
        #     x = self.model(x)
        #     output = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        # return output
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(
            x.size()[0], -1
        )  # consider other forward function

    def __downsampling_block(self, in_features: int, out_features: int) -> tuple:

        downsampling_block = (
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_features),  # try nn.BatchNorm2d()
            nn.LeakyReLU(0.2, inplace=True),
        )
        return downsampling_block
