import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class ConvBNRelu(nn.Module):
    def __init__(self,cin,cout,k=3,s=1):
        super(ConvBNRelu,self).__init__()
        if k==3:
            p=1
        elif k==1:
            p=0
        self.conv = nn.Sequential(
            nn.Conv2d(cin,cout,k,s,p),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,cin):
        super(ResBlock,self).__init__()
        mid = cin//2
        self.conv1x1 = nn.Conv2d(cin,cin,1,1,0)
        self.conv3x3 = nn.Conv2d(mid,mid,3,1,1)
        self.mid = mid
        self.bn = nn.BatchNorm2d(cin)

    def forward(self,x):
        x0 = self.conv1x1(x)
        x1 = x0[:,:self.mid,:,:]
        x2 = x0[:,self.mid:,:,:]
        x1 = self.conv3x3(x1)
        x3 = torch.cat([x1,x2],dim=1)
        x3 = self.bn(x3)
        x = x + x3
        return x

class ResNet(nn.Module):
    def __init__(self,repets=None,channels=None):
        super(ResNet,self).__init__()
        if repets==None:
            repets = [2,3,4,2]
        if channels==None:
            channels = [16,32,64,96,128]
        self.blocks = nn.ModuleList()
        tmp = ConvBNRelu(3,channels[0],k=3,s=2)
        self.blocks.append(tmp)
        for idx,repet in enumerate(repets):
            cin = channels[idx]
            cout = channels[idx+1]
            net = [ConvBNRelu(cin,cout,3,2)]
            for _ in range(repet):
                net.append(ResBlock(cout))
            net = nn.Sequential(*net)
            self.blocks.append(net)

    def forward(self,x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


def main():
    net = ResNet(repets=[2,3,4,2],channels=[16,32,64,96,128])
    from torchstat import stat
    stat(net,(3,448,768))


if __name__ == "__main__":
    main()