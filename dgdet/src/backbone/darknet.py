import torch
from torch import nn
from torch.nn import functional as F
from dgdet.src.opt.dropblock import DropBlock2D

class Conv3x3(nn.Module):
    def __init__(self,c_in,c_out,stride):
        super(Conv3x3,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.net(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,channel,dropblock=0.):
        super(ResBlock,self).__init__()
        net = [
            nn.Conv2d(channel,channel//2,kernel_size=1),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
            nn.Conv2d(channel//2,channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(channel),
        ]
        self.net = nn.Sequential(*net)
        self.dropout = DropBlock2D(drop_prob=dropblock,block_size=3)

    def forward(self,input):
        x = self.net(input)
        x = x+input
        x = F.relu(x)
        x = self.dropout(x)
        return x

class DarkStage(nn.Module):
    def __init__(self,c_in,c_out,repet,dropout=0.):
        super(DarkStage,self).__init__()
        self.down_smaple = nn.Conv2d(c_in,c_out,kernel_size=3,stride=2,padding=1)
        blocks = []
        for _ in range(repet):
            conv = ResBlock(c_out,dropout)
            blocks.append(conv)
        self.blocks = nn.Sequential(*blocks)

    def forward(self,x):
        x = self.down_smaple(x)
        x = self.blocks(x)
        return x

class DarkNet(nn.Module):
    def __init__(self,repets=None,channels=None,dropout=0.):
        super(DarkNet,self).__init__()
        if repets is None:
            repets = [2,3,4,2]
        if channels is None:
            channels = [16,24,48,72,108]
        cin = channels[0]
        # deep stream
        self.blocks = nn.ModuleList()
        conv0 = [
            Conv3x3(3,cin,1),
            DarkStage(cin,cin,1),
        ]
        conv0 = nn.Sequential(*conv0)
        self.blocks.append(conv0)
        for i in range(len(repets)):
            c_in = channels[i]
            c_out = channels[i+1]
            repet = repets[i]
            conv = DarkStage(c_in,c_out,repet,dropout=0.)
            if i!=0:
                conv = DarkStage(c_in,c_out,repet,dropout)
            self.blocks.append(conv)

    def forward(self,x):
        features = []
        for conv in self.blocks:
            x = conv(x)
            features.append(x)
        return features



def main():
    net = DarkNet(repets=[2,3,4,2],channels=[16,32,64,96,128])
    from torchstat import stat
    stat(net,(3,384,640))

if __name__ == "__main__":
    main()





    

