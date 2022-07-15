from turtle import forward
from matplotlib.pyplot import xlabel
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9) -> None:
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            # TODO if needed change conv2d and instancenorm arguments to
            # meet our needs
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_features, out_features, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            )
            in_features = out_features
            out_features = in_features * 2

    def forward(self, x):
        pass


class Discriminator(nn.Module):
    def __init__(self, input_nc, layers_number) -> None:
        super(Discriminator, self).__init__()

        # TODO implement number of layers regulation

        self.model = nn.Sequential(
            # 1st layer
            nn.Conv2d(input_nc, output_nc := 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd,
            nn.Conv2d(
                input_nc := output_nc,
                output_nc := output_nc * 2,
                4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(output_nc),  # try nn.BatchNorm2d()
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd,
            nn.Conv2d(
                input_nc := output_nc,
                output_nc := output_nc * 2,
                4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(output_nc),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th,
            nn.Conv2d(
                input_nc := output_nc,
                output_nc := output_nc * 2,
                4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(output_nc),
            nn.LeakyReLU(0.2, inplace=True),
            # classification layer
            nn.Conv2d(
                input_nc := output_nc,
                output_nc := 1,
                4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(
            x.size()[0], -1
        )  # consider other forward function
