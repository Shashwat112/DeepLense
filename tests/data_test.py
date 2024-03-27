import torch
from tests import experiment_used
if experiment_used == 'super_resolution':
    from tests import desired_dataloader_lr_shape, desired_dataloader_hr_shape
if experiment_used == 'diffusion':
    from tests import desired_dataloader_shape
from data.loader import *

class ShapeError(Exception):
    pass

def run():
    if experiment_used == 'super_resolution':
        x,y = next(iter(train_loader())),next(iter(test_loader()))

        if all([list(x[0].shape[1:]) == desired_dataloader_lr_shape, list(x[1].shape[1:]) == desired_dataloader_hr_shape]+
               [list(y[0].shape[1:]) == desired_dataloader_lr_shape, list(y[1].shape[1:]) == desired_dataloader_hr_shape]):
            if all([True if (i.dtype == torch.float32) and (j.dtype == torch.float32) else False for i,j in zip(x,y)]):
                print('Success')
            else:
                print('Failed\n')
                raise TypeError(f'Tensor type is not {torch.float32}')
        else:
            print('Failed\n')
            raise ShapeError('loader shape does not match desired shape')
    if experiment_used == 'diffusion':
        x = next(iter(train_loader(1)))

        if list(x.shape[1:]) == desired_dataloader_shape:
            if all([i.dtype == torch.float32 for i in x]):
                print('Success')
            else:
                print('Failed\n')
                raise TypeError(f'Tensor type is not {torch.float32}')
        else:
            print('Failed\n')
            raise ShapeError('loader shape does not match desired shape')
