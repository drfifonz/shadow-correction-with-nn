from torch.utils.data import DataLoader
from trainer import Trainer


def train(opt):
    """
    training model
    """
    Trainer(opt)

    if opt.resume:
        epoch_start = 3
    else:
        epoch_start = 0

    dataloader = DataLoader()

    # TRAINING
    print("Starting training loop...")
    for epoch in range(epoch_num := 3):
        for i, data_i in enumerate(dataloader, start=epoch_start):
            pass
