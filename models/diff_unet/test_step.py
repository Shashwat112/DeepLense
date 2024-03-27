import torch
from models import device
from models.utils import sv
from models.diff_unet.architecture import Model
from models.diff_unet import test_param
dev = torch.device('mps') if torch.backends.mps.is_available() and device=='gpu' else torch.device('cpu')

model = Model().to(dev)

@torch.no_grad()
def run(x: torch.Tensor, t: int, last: bool = False) -> torch.Tensor:
    if test_param.sampling_technique == 'ddpm':
        if last:
            return ((1/(sv('at',t))**0.5)*(x-model(x,torch.tensor(t).unsqueeze(0))*(sv('bt',t)/((1-sv('a_t',t))**0.5))))[0]
        return ((1/(sv('at',t))**0.5)*(x-model(x,torch.tensor(t).unsqueeze(0))*(sv('bt',t)/((1-sv('a_t',t))**0.5)))+sv('bt',t)*torch.randn_like(x))[0]
    if test_param.sampling_technique == 'ddim':
        rev = model(x,torch.tensor(t).unsqueeze(0))[0]
        return (1/(sv('at',t))**0.5)*(x-rev*((1-sv('a_t',t))**0.5))+rev*((1-sv('a_t',t-1))**0.5)
