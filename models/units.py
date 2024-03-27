import torch
from torch import nn
from models.utils import ptb

class ConvBlock(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,ker_size:int,stride:int,pad:int,maxpool:int|None=None,batch_norm:bool=True,group_norm:bool=False,act:nn=nn.ReLU())->None:
        super (ConvBlock,self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,ker_size,stride,pad)
        self.maxpool=nn.MaxPool2d(maxpool) if maxpool!=None else nn.Identity()
        self.bn=nn.BatchNorm2d(out_channel) if batch_norm else nn.Identity()
        self.gn=nn.GroupNorm(4,out_channel) if group_norm else nn.Identity()
        self.act=act
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.act(self.gn(self.bn(self.maxpool(self.conv(x)))))
    
class ConvTBlock(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,ker_size:int,stride:int,pad:int,out_pad:int=0,batch_norm:bool=True,group_norm:bool=False,act:nn=nn.ReLU())->None:
        super (ConvTBlock,self).__init__()
        self.conv=nn.ConvTranspose2d(in_channel,out_channel,ker_size,stride,pad,out_pad)
        self.bn=nn.BatchNorm2d(out_channel) if batch_norm else nn.Identity()
        self.gn=nn.GroupNorm(4,out_channel) if group_norm else nn.Identity()
        self.act=act
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.act(self.gn(self.bn(self.conv(x))))
    
class UpsampleBlock(nn.Module):
    def __init__(self,in_channel:int,scale_factor:int,out_channel:int=None,batch_norm:bool=True,act:nn=nn.ReLU())->None:
        super (UpsampleBlock,self).__init__()
        if out_channel==None:
            out_channel=in_channel
        self.upsample=nn.Sequential(nn.Conv2d(in_channel,in_channel*scale_factor**2,3,1,1),
                                    nn.PixelShuffle(scale_factor),
                                    nn.Conv2d(in_channel,out_channel,1,1,0))
        self.bn=nn.BatchNorm2d(in_channel) if batch_norm else nn.Identity()
        self.act=act
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.act(self.bn(self.upsample(x)))

class ResidualBlock(nn.Module):
    def __init__(self,no_of_convblocks:int,in_channel:int,batch_norm:bool=True,group_norm:bool=False,act:nn=nn.ReLU())->None:
        super (ResidualBlock,self).__init__()
        self.conv=nn.ModuleList([ConvBlock(in_channel,in_channel,3,1,1,batch_norm=batch_norm,group_norm=group_norm,act=act) for _ in range(no_of_convblocks)])
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return nn.Sequential(*self.conv)(x)+x

class TEmb(nn.Module):
    def __init__(self,out_ch:int,dim:int=512):
        super(TEmb,self).__init__()
        self.dim=dim
        self.lin=nn.Sequential(nn.Linear(dim,out_ch),
                               nn.SiLU())
    def forward(self,T:torch.Tensor):
        emb=torch.tensor([[torch.sin(T[j]/(10000**(2*i*(1/self.dim)))) if i%2==0 else torch.cos(T[j]/(10000**(2*i*(1/self.dim)))) for i in range(self.dim)] for j in range(len(T))])
        return self.lin(emb).unsqueeze(2).unsqueeze(3)

class Attention(nn.Module):
    def __init__(self,out_ch:int,nheads:int=8) -> None:
        super (Attention,self).__init__()
        self.attn=nn.MultiheadAttention(out_ch,num_heads=nheads,batch_first=True)
    def forward(self,x:torch.Tensor) -> torch.tensor:
        _x=ptb(x,1)
        out=_x.reshape(-1,_x.shape[1]*_x.shape[2],_x.shape[3])
        out=self.attn(out,out,out)[0]
        out=out.reshape(*_x.shape)
        return ptb(out,3)
    
class TEmbResBlock(nn.Module):
    def __init__(self,no_of_convblocks:int,in_channel:int,batch_norm:bool=True,group_norm:bool=False,act:nn=nn.ReLU())->None:
        super (TEmbResBlock,self).__init__()
        self.conv=nn.ModuleList([ConvBlock(in_channel,in_channel,3,1,1,batch_norm=batch_norm,group_norm=group_norm,act=act) for _ in range(no_of_convblocks)])
        self.temb=TEmb(in_channel)
    def forward(self,x:torch.Tensor,T:torch.Tensor)->torch.Tensor:
        out=self.conv[0](x)+self.temb(T)
        for module in self.conv[1:]:
            out=module(out)+self.temb(T)
        return out+x
    
class modSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs