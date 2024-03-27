import os
import torch
from models import device, ddp
from models.diff_unet.architecture import Model
from models.diff_unet import train_param as tp, timesteps as T
from engine import optimizer, weight_initializer
from engine.loss import mse
from models.utils import noising
from typing import Tuple

if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    dev = int(os.environ['LOCAL_RANK'])
else:
    dev = torch.device('mps') if torch.backends.mps.is_available() and device=='gpu' else torch.device('cpu')

model = Model().to(dev)
if ddp:
    model = DDP(model, device_ids=[dev])
weight_initializer.run(model, tp.weight_init, tp.bias_init)
opt = optimizer.run(model.parameters(), tp.optimizer, tp.learning_rate, tp.betas, tp.weight_decay)

def run(in_data: torch.Tensor) -> Tuple[torch.Tensor,float]:
    opt.zero_grad()
    t = torch.randint(1,T,(in_data.shape[0],))
    x_noisy, noise = noising(in_data, t)
    noise_pred = model(x_noisy.to(dev), t.to(dev))
    loss = mse(noise_pred, noise.to(dev))
    loss.backward()
    with torch.no_grad():
        opt.step()
    return loss.detach().item()