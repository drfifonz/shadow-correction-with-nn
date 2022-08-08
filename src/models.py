import torch
import torch.nn as nn
import torch.nn.functional as F


class Deshadower(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, res_blocks=9):
        super(Deshadower, self).__init__()

        # Initial conv layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # downsampling
        in_channels = 64  # to be changed
        out_channels = in_channels * 2

        downsampling = (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        for _ in range(2):
            self.model, *_ = map(self.model.append, downsampling)
            in_channels = out_channels
            out_channels = in_channels * 2

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
            raise Exception("res_blocks number should be greater than 0")

        for _ in range(res_blocks):
            self.model, *_ = map(self.model.append, residual_block)

        # upsampling
        out_channels = in_channels // 2
        upsampling = (
            nn.ConvTranspose2d(
                in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        for _ in range(2):
            self.model, *_ = map(self.model.append, upsampling)
            in_channels = out_channels
            out_channels = in_channels // 2

        # output layer
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(64, out_channels, 7))

    def forward(self, x):
        return (self.model(x) + x).tanh()


class Shadower(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, res_blocks=9):
        super(Shadower, self).__init__()

        # initial conv layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            # Additional channel is added to in_channels
            # because of the presence of the mask
            nn.Conv2d(in_channels + 1, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # downsampling
        in_channels = 64
        out_channels = in_channels * 2
        downsampling = (
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        for _ in range(2):
            self.model, *_ = map(self.model.append, downsampling)
            in_channels = out_channels
            out_channels = in_channels * 2

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
            raise Exception("res_blocks number should be greater than 0")

        for _ in range(res_blocks):
            self.model, *_ = map(self.model.append, residual_block)

        # upsampling
        out_channels = in_channels // 2
        upsampling = (
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for _ in range(2):
            self.model, *_ = map(self.model.append, upsampling)

        in_channels = out_channels
        out_channels = in_channels // 2

        # output layer
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(64, out_channels, 7))

    def forward(self, x, mask):
        return (self.model(torch.cat((x, mask), 1)) + x).tanh()


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, layers_number=3) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels := 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        downsampling = (
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
            self.model, *_ = map(self.model.append, downsampling)

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

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(
            x.size()[0], -1
        )  # consider other forward function
