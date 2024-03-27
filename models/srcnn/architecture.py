import torch
from torch import nn 
from models.units import ConvBlock, ConvTBlock
from models import hr_shape
from models.srcnn import encoder_channel_sequence as ecs, decoder_channel_sequence as dcs

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.encoder = nn.ModuleList([ConvBlock(in_ch,out_ch,3,2,1,batch_norm=False) for in_ch,out_ch in zip(ecs[:-1],ecs[1:])])
        self.decoder = nn.ModuleList([ConvTBlock(in_ch,out_ch,4-i%2,2,1,batch_norm=False) for i,(in_ch,out_ch) in enumerate(zip(dcs[:-1],dcs[1:]))]+
                                     [nn.AdaptiveAvgPool2d(hr_shape[1:])])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.Sequential(*self.encoder,*self.decoder)(x)
    
