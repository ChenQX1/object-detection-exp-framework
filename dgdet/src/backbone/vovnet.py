import torch
from torch import nn
from torch.nn import functional as F

class ConvBNRelu(nn.Module):
    def __init__(self,c_in,c_out,stride=1):
        super(ConvBNRelu,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.net(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self,cin,cout,repet=3,channel=32):
        super(BottleNeck,self).__init__()
        self.convs = nn.ModuleList()
        for idx in range(repet):
            if idx==0:
                tmp = ConvBNRelu(cin,channel)
            else:
                tmp = ConvBNRelu(channel,channel)
            self.convs.append(tmp)
        cmid = cin+repet*channel
        self.conv1x1 = nn.Conv2d(cmid,cout,1,1,0)

    def forward(self,x):
        feats = [x]
        for conv in self.convs:
            x = conv(x)
            feats.append(x)
        x = torch.cat(feats,dim=1)
        x = self.conv1x1(x)
        return x

class DownSample(nn.Module):
    def __init__(self,cin):
        super(DownSample,self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.conv1x1 = nn.Conv2d(cin,cin,1,1,0)

    def forward(self,x):
        x = self.pool(x)
        x = self.conv1x1(x)
        return x


class VovNet(nn.Module):
    def __init__(self,repets=None,channels=None,g=32):
        super(VovNet,self).__init__()
        if repets==None:
            repets = [2,2,2,2]
        if channels==None:
            channels = [16,32,64,128,192] 
        self.blocks = nn.ModuleList()
        self.blocks.append(ConvBNRelu(3,channels[0],2))
        for idx,repet in enumerate(repets):
            cin = channels[idx]
            cout = channels[idx+1]
            down = DownSample(cin)
            tmp = BottleNeck(cin,cout,repet,g)
            tmp = nn.Sequential(down,tmp)
            self.blocks.append(tmp)

    def forward(self,x):
        feats = []
        for net in self.blocks:
            #print(x.shape)
            x = net(x)
            feats.append(x)
        return feats

def main():
    net = VovNet(repets=[2,3,3,3],channels=[16,48,96,128,192])
    net.eval()
    x = torch.rand(1,3,384,640)
    data = net(x)
    for item in data:
        print(item.shape)
    from torchstat import stat
    stat(net,(3,384,640))


if __name__ == "__main__":
    main()
            
        
        










    

