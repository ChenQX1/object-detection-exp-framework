import torch
from torch import nn
from mmcv.ops import DeformConv2dPack


class RepetHead(nn.Module):
    def __init__(self,cin,repet):
        super(RepetHead,self).__init__()
        net = []
        for i in range(repet):
            tmp = DeformConv2dPack(cin,cin,3,1,1)
            net.append(tmp)
            net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


def main():
    net = RepetHead(64,3)
    from torchstat import stat
    stat(net,(64,120,120))

if __name__ == "__main__":
    main()