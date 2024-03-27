import torch
import numpy as np
from typing import Tuple
from models import exp_name
if exp_name == 'diffusion':
    from models.diff_unet import noise_sched, timesteps as T

# Switches from (B,H,W,C)=>(B,C,H,W) and vice-versa
def ptb(x: torch.Tensor, ch_dim: int) -> torch.Tensor:
    return (torch.transpose(torch.transpose(x,2,3),1,2) if len(x.shape)==4 else torch.transpose(torch.transpose(x,1,2),0,1)) if ch_dim==2 or ch_dim==3 else (torch.transpose(torch.transpose(x,1,2),2,3) if len(x.shape)==4 else torch.transpose(torch.transpose(x,0,1),1,2))

# Schedule variables
def sv(str: str, t: int) -> torch.Tensor:

    if noise_sched == 'cosine':
        ft=lambda t,s:np.cos((((t/T)+s)/(1+s))*(np.pi/2))**2
        a_t=ft(t,0.008)/ft(0,0.008)
        bt=1-a_t/(ft(t-1,0.008)/ft(0,0.008))
        bt=0.02 if bt>0.02 else bt
        at=1-bt

    if noise_sched == 'linear':
        bt=torch.linspace(0.0001,0.02,T)[t-1]
        at=1-bt
        a_t=torch.cumprod(1-torch.linspace(0.0001,0.02,T),dim=-1)[t-1]

    return bt if str=='bt' else (at if str=='at' else (a_t if str=='a_t' else 'nan'))

# Noising function
def noising(x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    noise=torch.randn_like(x)
    return (torch.stack([(sv('a_t',t[i])**0.5)*x[i]+((1-sv('a_t',t[i]))**0.5)*noise[i] for i in range(len(t))]),noise)