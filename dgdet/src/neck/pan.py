import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self,cin,k,s,p):
        super(Conv,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin,cin,k,s,p),
            nn.BatchNorm2d(cin),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.net(x)
        return x

class PAN(nn.Module):
    def __init__(self,cin,nb_fpn=4):
        super(PAN,self).__init__()
        self.fpn = nb_fpn-1
        self.conv3x3_s1_p0 = Conv(cin,3,1,1)
        self.conv3x3_s2 = nn.ModuleList()
        self.conv3x3_s1 = nn.ModuleList()
        for _ in range(self.fpn):
            self.conv3x3_s2.append(Conv(cin,3,2,1))
            self.conv3x3_s1.append(Conv(cin,3,1,1))

    def forward(self,x):
        x0 = self.conv3x3_s1_p0(x[0])
        feat_n = [x0]
        for i in range(self.fpn):
            tmp = self.conv3x3_s2[i](feat_n[i])
            tmp = tmp + x[i+1]
            tmp = self.conv3x3_s1[i](tmp)
            feat_n.append(tmp)
        return feat_n
