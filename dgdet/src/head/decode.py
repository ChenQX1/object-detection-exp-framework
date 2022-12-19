import torch
from torch import nn
import torch.nn.functional as F

class ClfHead(nn.Module):
    def __init__(self,cin,num_anchors=1,num_classes=1):
        super(ClfHead,self).__init__()
        self.deploying = False
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        cout = self.num_anchors*self.num_classes
        self.conv = nn.Conv2d(cin,cout,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.conv(x)
        if self.deploying:
            return x
        out1 = x.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out2 = out2.contiguous().view(x.shape[0], -1, self.num_classes)
        return out2

class RegHead(nn.Module):
    def __init__(self,cin,num_anchors):
        super(RegHead,self).__init__()
        self.deploying = False
        cout = num_anchors*4
        self.conv = nn.Conv2d(cin,cout,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.conv(x)
        if self.deploying:
            return x
        # out is B x C x W x H, with C = 4*num_anchors
        out = x.permute(0, 2, 3, 1)
        # out:(bs,A,4)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClfHead1x1(nn.Module):
    def __init__(self,cin,num_anchors=1,num_classes=1):
        super(ClfHead1x1,self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        cout = self.num_anchors*self.num_classes
        self.conv = nn.Conv2d(cin,cout,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv(x)
        out1 = x.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out2 = out2.contiguous().view(x.shape[0], -1, self.num_classes)
        return out2

class RegHead1x1(nn.Module):
    def __init__(self,cin,num_anchors):
        super(RegHead1x1,self).__init__()
        cout = num_anchors*4
        self.conv = nn.Conv2d(cin,cout,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv(x)
        # out is B x C x W x H, with C = 4*num_anchors
        out = x.permute(0, 2, 3, 1)
        # out:(bs,A,4)
        return out.contiguous().view(out.shape[0], -1, 4)