import torch
from torch import nn
from torchvision.models import resnet
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN


def _resnet(backbone_name='resnet50',pretrained=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
    return backbone

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        backbone = _resnet()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1,x2,x3,x4]

def main():
    net = ResNet()
    data = torch.randn(1,3,384,640)
    feats = net(data)
    for item in feats:
        print(item.shape)
    

if __name__ == "__main__":
    main()