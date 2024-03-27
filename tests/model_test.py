import models
import pkgutil
from tests import *
import torch

class RegisterError(Exception):
    pass

class ShapeError(Exception):
    pass

class models_registered:
    def __init__(self) -> None:
        pass
    
    def srcnn(self) -> bool:
        from models.srcnn.architecture import Model as srcnn
        sample_in_srcnn = torch.randn(tuple(desired_dataloader_lr_shape)).unsqueeze(0)
        sample_out_srcnn = srcnn()(sample_in_srcnn)
        return list(sample_out_srcnn.squeeze(0).shape) == desired_model_output_shape
    
    def srgan(self) -> bool:
        from models.srgan.architecture import Model as srgan
        sample_in_gen = torch.randn(tuple(desired_dataloader_lr_shape)).unsqueeze(0)
        sample_in_dis = torch.randn(tuple(desired_dataloader_hr_shape)).unsqueeze(0)
        sample_out_gen = srgan.Generator()(sample_in_gen)
        sample_out_dis = srgan.Discriminator()(sample_in_dis)
        return (list(sample_out_gen.squeeze(0).shape) == desired_generator_output_shape) and (list(sample_out_dis.squeeze(0).shape) == desired_discriminator_output_shape)
    
    def diff_unet(self) -> bool:
        from models.diff_unet.architecture import Model as diff
        sample_in_diff_unet = (torch.randn(tuple(desired_dataloader_shape)).unsqueeze(0), torch.randint(0,100,(1,)))
        sample_out_diff_unet = diff()(*sample_in_diff_unet)
        return list(sample_out_diff_unet.squeeze(0).shape) == desired_model_output_shape

def run():
    obj = models_registered()
    model_pkges = [name for _,name,ispkg in pkgutil.iter_modules(models.__path__) if ispkg]
    if all([True if name in model_register else False for name in model_pkges]):
        if len([m for m in dir(models_registered) if not (m.startswith('__') and m.endswith('__'))]) == len(model_pkges):
            if getattr(obj,model_used)():
                print('Success')
            else:
                print('Failed\n')
                raise ShapeError('Model output shape does not match desired shape')
        else:
            print('Failed\n')
            raise ModuleNotFoundError(f'Test functions for one or more models missing')
    else:
        print('Failed\n')
        raise RegisterError('Model does not exist in register')
