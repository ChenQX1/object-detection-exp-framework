import torch
from torch import nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self,cin,cout,k,s,p):
        super(ConvBnRelu,self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(cin,cout,k,s,p),
            nn.BatchNorm2d(cin),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.nn(x)

'''
replace ConvBnRelu3x3 by 1x1 and slice 3x3
'''
class Conv3x3(nn.Module):
    def __init__(self,cin,factor=0.5):
        super(Conv3x3,self).__init__()
        self.conv1x1 = ConvBnRelu(cin,cin,1,1,0)
        self.p1 = int(cin*factor)
        self.p2 = cin-self.p1
        self.conv3x3 = ConvBnRelu(self.p1,self.p1,3,1,1)

    def forward(self,x):
        x = self.conv1x1(x)
        x1,x2 = torch.split(x,[self.p1,self.p2],dim=1)
        x1 = self.conv3x3(x1)
        x3 = torch.cat([x1,x2],dim=1)
        return x3

class RepetHead(nn.Module):
    def __init__(self,cin,repet=1,f=0.25):
        super(RepetHead,self).__init__()
        part = []
        for _ in range(repet):
            part.append(Conv3x3(cin,f))
        self.part = nn.Sequential(*part)

    def forward(self,x):
        return self.part(x)


def main():
    net1 = RepetHead(64,repet=2)
    net2 = nn.Sequential(
        nn.Conv2d(64,64,3,1,1),
        nn.Conv2d(64,64,3,1,1),
    )
    from torchstat import stat
    stat(net1,(64,96,160))


if __name__ == "__main__":
    main()