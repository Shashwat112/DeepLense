import torch
from torchvision.transforms import Compose,Lambda,functional

transforms= Compose([torch.Tensor,
                    Lambda(lambda x:functional.convert_image_dtype(x,torch.float32))])

