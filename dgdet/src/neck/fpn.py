from torch import nn
from torch.nn import functional as F
import torch

class FPN(nn.Module):
    def __init__(self,channels,cout=128,down=1):
        super(FPN,self).__init__()
        #define conv for down sample
        self.down_sampe_conv = nn.ModuleList()
        for i in range(down):
            if i==0:
                conv = nn.Conv2d(channels[-1],cout,kernel_size=3,padding=1,stride=2)
            else:
                conv = nn.Conv2d(cout,cout,kernel_size=3,padding=1,stride=2)
            self.down_sampe_conv.append(conv)
        
        #define 1x1 convs for squeeze featmaps
        self.squeeze_conv = nn.ModuleList()
        #
        channels.reverse() 
        for channel in channels:
            conv = nn.Conv2d(channel,cout,kernel_size=1,stride=1)
            self.squeeze_conv.append(conv)

    def upsample(self,x,scale_factor):
        x = F.interpolate(x,scale_factor=scale_factor,mode='bilinear',align_corners=True)
        return x

    def forward(self,feature_maps):
        #down sample
        feat_down = []
        x = feature_maps[-1]
        for block in self.down_sampe_conv:
            x = block(x)
            feat_down.append(x)
        #up sample
        feature_maps.reverse()
        count = 0
        feat_up = []
        for x,block in zip(feature_maps,self.squeeze_conv):
            x = block(x)
            if count!=0:
                #x += F.interpolate(feat_up[-1],scale_factor=2.,mode='bilinear',align_corners=True)
                x += self.upsample(feat_up[-1],scale_factor=2)
            feat_up.append(x)
            count+=1
        feat_up.reverse() # from p2 to p5
        feat_up.extend(feat_down)  # add p6 - p7
        return feat_up