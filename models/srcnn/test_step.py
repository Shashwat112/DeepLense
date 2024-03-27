import torch
from models import device
from models.srcnn.architecture import Model
dev = torch.device('mps') if torch.backends.mps.is_available() and device=='gpu' else torch.device('cpu')

model = Model().to(dev)

@torch.no_grad()
def run(test_data: torch.Tensor) -> torch.Tensor:
    pred = model(test_data.to(dev))
    return pred.detach()