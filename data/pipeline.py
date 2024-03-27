import torch
from torch.utils.data import Dataset, random_split
from data import exp_name
if exp_name == 'super_resolution':
    from data import train_test_split
from data.source import cont_data
from data.utils import vision_2_vision
import numpy as np
from typing import Tuple

data_split_seed = 42

class sr_pipe(Dataset):
    def __init__(self, loader_type: str) -> None:
        self.full_data = cont_data
        self.trainset, self.testset = random_split(self.full_data,train_test_split,torch.Generator().manual_seed(data_split_seed))
        if loader_type == 'train':
            self.loader = self.trainset
        elif loader_type=='test':
            self.loader=self.testset
        else:
            raise Exception('loader_type expects only \'train\' or \'test\' values')
        
    def __len__(self) -> int:
        return len(self.loader)
    
    def __getitem__(self,i:int) -> Tuple[torch.Tensor,torch.Tensor]:
        return vision_2_vision(self.loader[i])
    
class diff_pipe(Dataset):
    def __init__(self) -> None:
        self.full_data: np.ndarray = cont_data

    def __len__(self) -> int:
        return len(self.full_data)
    
    def __getitem__(self,i) -> torch.Tensor:
        return vision_2_vision(self.full_data[i])
    

    

