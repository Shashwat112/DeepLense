import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

net = vgg19(weights=VGG19_Weights.DEFAULT).features[:-1]
for m in net.parameters():
    m.requires_grad=False

def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = nn.MSELoss()
    return loss(x,y)

def bce(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = nn.BCELoss()
    return loss(x,y)

def nll(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = nn.NLLLoss()
    return loss(x,y)

def vgg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = nn.MSELoss()
    x_, y_ = nn.functional.conv2d(x,torch.ones((3,1,1,1)).to(x.device)), nn.functional.conv2d(y,torch.ones((3,1,1,1)).to(y.device))
    loss = mse(net(x_),net(y_))
    return loss




