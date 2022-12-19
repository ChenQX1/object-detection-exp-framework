from dgdet.detector.retinanet_p3_7 import RetinaNet
from torchstat import stat
import torch

def main():
    net = RetinaNet()
    net.eval()
    #torch.save(net.state_dict(),'weight/test.pth')
    net.anchors.use_cuda = False
    net.output.use_cuda = False
    shape = (3,448,768)
    #shape = (3,448,768)
    stat(net,shape)
    print(net.anchors.pyramid_levels)
    print(net.anchors.sizes)


if __name__ == "__main__":
    main()
