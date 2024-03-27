import torch
from torch import nn 
from models.units import ConvBlock, UpsampleBlock, ResidualBlock
from models import hr_shape
from models.srgan import discriminator_channel_sequence as dcs, generator_residual_blocks as grb, residual_channel_number as rcn, upsample_architecture as us

class Model:
    def __init__(self) -> None:
        pass
    class Discriminator(nn.Module):

        def __init__(self) -> None:
            super (Model.Discriminator, self).__init__()
            self.conv=nn.ModuleList([ConvBlock(dcs[0],dcs[1],3,1,1,batch_norm=False,act=nn.LeakyReLU(0.2))]+
                                    [ConvBlock(in_ch,out_ch,3,2-i%2,1,act=nn.LeakyReLU(0.2)) for i,(in_ch,out_ch) in enumerate(zip(dcs[1:-1],dcs[2:]))]+
                                    [nn.AdaptiveAvgPool2d(6)])
            self.lin=nn.Sequential(nn.Flatten(),
                                   nn.Linear(512*6*6,64),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(64,1),
                                   nn.Sigmoid())
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nn.Sequential(*self.conv,self.lin)(x)
        
    class Generator(nn.Module):

        def __init__(self) -> None:
            super (Model.Generator, self).__init__()
            self.conv=ConvBlock(1,rcn,9,1,1,batch_norm=False,act=nn.PReLU(rcn))
            self.residual=nn.ModuleList([ResidualBlock(no_of_convblocks=2,in_channel=rcn,act=nn.PReLU(rcn)) for _ in range(grb)]+
                                        [ConvBlock(rcn,rcn,3,1,1,act=nn.Identity())])
            self.upsample=nn.ModuleList([UpsampleBlock(rcn,2,us[0],batch_norm=False,act=nn.PReLU(us[0])),
                                         UpsampleBlock(us[0],2,us[1],batch_norm=False,act=nn.PReLU(us[1])),
                                         ConvBlock(us[1],us[2],9,1,1,batch_norm=False,act=nn.Identity()),
                                         nn.AdaptiveAvgPool2d(hr_shape[1:])])
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out=self.conv(x)
            out=out+nn.Sequential(*self.residual)(out)
            return nn.Sequential(*self.upsample)(out)