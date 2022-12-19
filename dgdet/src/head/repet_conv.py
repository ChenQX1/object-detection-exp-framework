import torch
from torch import nn
import torch.nn.functional as F

class RepetHead(nn.Module):
    def __init__(self,c_in,c_out,repet=1):
        super(RepetHead,self).__init__()
        conv0 = nn.Conv2d(c_in,c_out,kernel_size=3,padding=1)
        self.conv = [conv0,nn.ReLU()]
        if repet>1:
            for _ in range(repet-1):
                self.conv.append(nn.Conv2d(c_out,c_out,3,1,1))
                self.conv.append(nn.ReLU())
        self.conv = nn.Sequential(*self.conv)

    def forward(self,x):
        return self.conv(x)

class Repet1x1(nn.Module):
    def __init__(self,cin,repet):
        super(Repet1x1,self).__init__()
        net = []
        for i in range(repet):
            net.append(nn.Conv2d(cin,cin,1,1,0))
            if i!=repet:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        self.bn = nn.BatchNorm2d(cin)
    
    def forward(self,x):
        x = self.net(x)
        x = self.bn(x)
        return x

