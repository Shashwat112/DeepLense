import torch
from torch import nn
from models.units import TEmb, Attention, TEmbResBlock, ConvBlock, ConvTBlock, modSequential

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = ConvBlock(1,128,3,1,1,batch_norm=False,act=nn.SiLU())
        self.t1 = TEmb(128)
        self.resd1 = modSequential(TEmbResBlock(2,128,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                  ConvBlock(128,256,3,2,1,group_norm=True,act=nn.SiLU()))
        self.t2 = TEmb(256)
        self.resd2 = modSequential(TEmbResBlock(2,256,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                  ConvBlock(256,512,3,2,1,group_norm=True,act=nn.SiLU()))
        self.t3 = TEmb(512)
        self.resd3 = modSequential(TEmbResBlock(2,512,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                  Attention(512),
                                  ConvBlock(512,1024,3,2,1,group_norm=True,act=nn.SiLU()))
        self.t4 = TEmb(1024)
        self.resm1 = TEmbResBlock(2,1024,batch_norm=False,group_norm=True,act=nn.SiLU())
        self.resm2 = modSequential(TEmbResBlock(2,1024,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                 Attention(1024))
        self.t5 = TEmb(1024)
        self.resu3 = modSequential(TEmbResBlock(2,1024,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                  Attention(1024),
                                  ConvTBlock(1024,512,4,2,1,group_norm=True,act=nn.SiLU()))
        self.t6 = TEmb(512)
        self.resu2 = modSequential(TEmbResBlock(2,512,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                  Attention(512),
                                  ConvTBlock(512,256,3,2,1,group_norm=True,act=nn.SiLU()))
        self.t7 = TEmb(256)
        self.resu1 = modSequential(TEmbResBlock(2,256,batch_norm=False,group_norm=True,act=nn.SiLU()),
                                  ConvTBlock(256,128,4,2,1,group_norm=True,act=nn.SiLU()))
        self.t8 = TEmb(128)
        self.conv2 = ConvBlock(128,1,3,1,1,batch_norm=False,act=nn.Identity())

    def forward(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        out0 = self.conv1(x) + self.t1(T)               # (128x150x150)
        out1 = self.resd1(out0,T) + self.t2(T)          # (256x75x75)
        out2 = self.resd2(out1,T) + self.t3(T)          # (512x38x38)
        out3 = self.resd3(out2,T) + self.t4(T)          # (1024x19x19)
        out4 = self.resm1(out3,T)                       # (1024x19x19)
        out5 = self.resm2(out4,T) + self.t5(T)          # (1024x19x19)
        out6 = self.resu3(out5,T) + self.t6(T) + out2   # (512x38x38)
        out7 = self.resu2(out6,T) + self.t7(T) + out1   # (256x75x75)
        out8 = self.resu1(out7,T) + self.t8(T) + out0   # (128x150x150)
        return self.conv2(out8)                         # (1x150x150)