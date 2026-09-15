from __future__ import annotations
import torch
from torch import nn

class Block(nn.Sequential):
    def __init__(self, a: int, b: int):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())

class TinyUNet(nn.Module):
    """Three-level U-Net with GroupNorm; no pretrained weights."""
    def __init__(self, width: int = 12):
        super().__init__()
        self.e1=Block(1,width); self.e2=Block(width,width*2); self.e3=Block(width*2,width*4)
        self.bridge=Block(width*4,width*8); self.pool=nn.MaxPool2d(2)
        self.d3=Block(width*12,width*4); self.d2=Block(width*6,width*2); self.d1=Block(width*3,width)
        self.out=nn.Conv2d(width,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); z=self.bridge(self.pool(c))
        for skip, block in ((c,self.d3),(b,self.d2),(a,self.d1)):
            z=nn.functional.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            z=block(torch.cat((z,skip),dim=1))
        return self.out(z)
