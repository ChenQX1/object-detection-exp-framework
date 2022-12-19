import torch
from torch import log, mode, nn
import torch.nn.functional as F

class MergeConv(nn.Module):
    def __init__(self,high,low):
        super(MergeConv,self).__init__()
        self.conv = nn.Conv2d(high,low,1,1,0)
    
    def forward(self,f1,f2):
        f2 = self.conv(f2)
        f2 = F.interpolate(f2,scale_factor=2)
        f = f1 + f2
        return f

class LFPN(nn.Module):
    def __init__(self,channels,cout=128,down=1):
        super(LFPN,self).__init__()
        # init merge conv
        self.merge_conv = nn.ModuleList()
        for i in range(len(channels)-1):
            conv = MergeConv(channels[i+1],channels[i])
            self.merge_conv.append(conv)
        # init down conv
        self.down_conv = nn.ModuleList()
        for i in range(down):
            if i==0:
                conv = nn.Conv2d(channels[-1],cout,3,2,1)
            else:
                conv = nn.Conv2d(cout,cout,3,2,1)
            self.down_conv.append(conv)
        # init conv3x3
        self.conv3x3s = nn.ModuleList()
        for c in channels:
            self.conv3x3s.append(nn.Sequential(
                nn.Conv2d(c,cout,3,1,1),
                nn.BatchNorm2d(cout),
            ))

    def forward(self,features):
        down = []
        x = features[-1]
        for conv in self.down_conv:
            x = conv(x)
            down.append(x)
        up = []
        features.reverse() #top-down
        for i,f in enumerate(features):
            if i>0:
                last = features[i-1]
                f = self.merge_conv[-i](f,last)
            f = self.conv3x3s[-(i+1)](f)
            up.append(f)
        up.reverse()
        up.extend(down)
        return up

def main():
    s = 160
    channel = [16,32,64,128]
    model = LFPN(channel,64,1)
    f = []
    for c in channel:
        f.append(torch.randn(1,c,s,s))
        s //= 2
    # for item in f:
    #     print(item.shape)
    logit = model(f)
    #print('output')
    for item in logit:
        print(item.shape)

if __name__ == "__main__":
    main()




        