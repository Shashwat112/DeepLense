import os
import torch
from models import device, ddp
from models.srcnn.architecture import Model
from models.srcnn import train_param as tp
from engine import optimizer, weight_initializer
from engine.loss import mse
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

def run(in_data: torch.Tensor, lab_data: torch.Tensor) -> Tuple[torch.Tensor,float]:
    y_pred = model(in_data.to(dev))
    loss = mse(y_pred,lab_data.to(dev))
    loss.backward()
    with torch.no_grad():
        opt.step()
        opt.zero_grad()
    return y_pred.detach()