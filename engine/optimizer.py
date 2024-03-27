from torch import optim
from typing import Iterator as itr, List

def run(model_param: itr, method: str, lr: float, betas: List[int], weight_decay: float):
    return getattr(optim,method)(model_param,lr=lr,betas=betas,weight_decay=weight_decay)
