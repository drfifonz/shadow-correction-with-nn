import sys

sys.path.insert(1, "./src")

import pytest

import trainer

from utils.arguments_parser import arguments_parser


"""
To run tests type:
pytest 
"""


@pytest.fixture
def generator_parameters():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def init_options():
    opt = arguments_parser()
    trainer_object = trainer.Trainer(opt)
    return opt, trainer_object


# @pytest.mark.skip("skipping init test")
def test_trainer_init():
    opt = arguments_parser()
    assert trainer.Trainer(opt)


# @pytest.mark.usefixtures("generator_parameters")
def test_run_one_batch_gen(generator_parameters=[1, 2, 3, 4, 5]):
    print(*generator_parameters)
    opt = arguments_parser()
    trainer_object = trainer.Trainer(opt)

    assert trainer_object.run_one_batch_for_generator(*[1, 2, 3, 4, 5])


# @pytest.mark.usefixtures("init_options")
def test_optimizers_init(init_options):
    # opt, trainer_object = init_options
    # trainer_object = trainer.Trainer(opt)

    opt = arguments_parser()
    trainer_object = trainer.Trainer(opt)

    assert trainer_object.optimizers_init()


@pytest.mark.usefixtures("init_options")
def test_learning_rate_schedulers_init(init_options):
    opt, trainer_object = init_options
    # trainer_object = trainer.Trainer(opt)
    assert trainer_object.learning_rate_schedulers_init(opt, 1)
