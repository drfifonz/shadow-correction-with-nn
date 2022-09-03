import sys

sys.path.insert(1, "./src")

import pytest
import torch

from utils import visualizer

from utils.arguments_parser import arguments_parser

"""
To run tests type:
pytest 
"""


@pytest.fixture()
def images():
    t1 = torch.Tensor(1, 2, 3)
    t2 = torch.Tensor(4, 5, 6)
    return [t1, t2]


@pytest.fixture()
def root_path():
    return "./src"


@pytest.fixture
def init_options():
    opt = arguments_parser()
    visualizer_object = visualizer.Visualizer(opt)
    return opt, visualizer_object


@pytest.mark.usefixtures("init_options")
@pytest.mark.usefixtures("root_path")
@pytest.mark.usefixtures("images")
def test_save_image():
    vis, opt = init_options
    assert vis.save_images(images, root_path)
