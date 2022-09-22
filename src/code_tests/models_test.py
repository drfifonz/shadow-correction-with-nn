import sys

sys.path.insert(1, "./src")

import models
import pytest


@pytest.fixture()
def channels():
    return 3, 64


def test_deshadower(channels):
    in_channels, out_channels = channels
    assert models.Deshadower1(in_channels, out_channels)


def test_shadower(channels):
    in_channels, out_channels = channels
    assert models.Shadower(in_channels, out_channels)


def test_discriminator(channels):
    in_channels, _ = channels
    assert models.Discriminator(
        in_channels,
    )
