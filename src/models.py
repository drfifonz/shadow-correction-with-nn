from turtle import forward
from matplotlib.pyplot import xlabel
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO class name to be changed
# TODO push models.py again


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock).__init__()

        convolution_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )
        self.conv_block = nn.Sequential(*convolution_block)

    def forward(self, x):
        return x + self.conv_block(x)


# former Generator class
class Deshadower(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks=9):
        super(Deshadower, self).__init__()

        # Initial conv layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            # TODO if needed change conv2d and instancenorm arguments to
            # meet our needs
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # downsampling
        in_channels = 64
        out_channels = in_channels * 2

        for _ in range(2):
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            in_channels = out_channels
            out_channels = in_channels * 2

        # temporary solution to be changed later
        model = []

        # residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_channels)]

        # adding residual blocks
        self.model = nn.Sequential(*model)

        # upsampling
        out_channels = in_channels // 2  # floor division
        for _ in range(2):
            self.model += nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            in_channels = out_channels
            out_channels = in_channels // 2

        # output layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7)
        )

    def forward(self, x):
        return (self.model(x) + x).tanh()


# work-in-progress name
class Shadower(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks=9):
        super(Shadower, self).__init__()

        # initial conv layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels + 1, 64, 7),  # + mask
            nn.InstanceNorm2d(64),
            nn.ReLu(inplace=True),
        )

        # downsampling
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        in_channels = out_channels
        out_channels = in_channels * 2

        # residual blocks
        model = []
        for _ in range(res_blocks):
            model += [ResidualBlock(in_channels)]

        self.model = nn.Sequential(*model)

        # upsampling
        out_channels = in_channels // 2
        for _ in range(2):
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        in_channels = out_channels
        out_channels = in_channels // 2

        # output layer
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7)
        )

    def forward(self, x, mask):
        return (self.model(torch.cat((x, mask), 1)) + x).tanh()


class Discriminator(nn.Module):
    def __init__(self, input_nc, layers_number=3) -> None:
        super(Discriminator, self).__init__()

        # TODO implement number of layers regulation

        # 1st layer
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, output_nc := 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2nd
        for _ in range(layers_number):
            self.model = nn.Sequential(
                nn.Conv2d(
                    input_nc := output_nc,
                    output_nc := output_nc * 2,
                    4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(output_nc),  # try nn.BatchNorm2d()
                nn.LeakyReLU(0.2, inplace=True),
            )

        # classification layer
        self.model = nn.Conv2d(
            input_nc := output_nc,
            output_nc := 1,
            4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(
            x.size()[0], -1
        )  # consider other forward function
