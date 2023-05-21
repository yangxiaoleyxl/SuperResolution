from torch import nn


def loss_fn(loss_name):
    assert (loss_name in ["MSE", "L1"])
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == "L1":
        return nn.L1Loss()
