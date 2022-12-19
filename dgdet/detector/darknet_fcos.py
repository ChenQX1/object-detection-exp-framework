import math
import torch
from torch import nn
from dgdet.src import head
from dgdet.src import loss
from dgdet.src import assign
from dgdet.src.backbone.darknet import DarkNet
from dgdet.src.neck.fpn import FPN
from dgdet.src.utils.anchor import Anchors
from dgdet.src.head.part_conv import RepetHead
from dgdet.src.head import decode
from dgdet.src.post_process.sigmoid import OutputLayer

class RetinaNet(nn.Module):
    def __init__(self):
        super(RetinaNet,self).__init__()
        #define encode 
        self.backbone = DarkNet(repets=[2,3,4,2],channels=[16,24,48,72,108])
        self.neck = FPN([24,48,72,108],cout=48,down=1)
        self.head = nn.ModuleList()
        for _ in range(5):
            self.head.append(RepetHead(48,2,f=0.25))
        #define decode 
        self.anchors = Anchors(
            pyramid_levels = [2, 3, 4, 5, 6],
            sizes = [1,1,1,1,1],
        )
        self.classificationModel = decode.ClfHead(48,num_anchors=1,num_classes=1)
        self.regressionModel = nn.ModuleList()
        for _ in range(5):
            self.regressionModel.append(decode.RegHead(48,num_anchors=1))
        self.output = OutputLayer()
        #define loss
        st_assign = assign.FCOS_Assign(scales=[32,64,128,256,512])
        clf_loss = loss.FocalLoss()
        reg_loss = loss.GIou_ltrb()
        self.loss_head = loss.LossHead(clf_loss,reg_loss,st_assign)
    
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def init_weight(self):
        #init clf head
        prior = 0.01
        self.classificationModel.conv.weight.data.fill_(0)
        self.classificationModel.conv.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #init reg head
        self.regressionModel.conv.weight.data.fill_(0)
        self.regressionModel.conv.bias.data.fill_(0)

    def forward(self,inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        features = self.backbone(img_batch)
        features = self.neck(features[1:])
        #features = [self.head(x) for x in features]
        features2 = []
        for feat,head in zip(features,self.head):
            tmp = head(feat)
            features2.append(tmp)

        classification = torch.cat([self.classificationModel(feature) for feature in features2], dim=1)
        #regression = torch.cat([self.regressionModel(feature) for feature in features2], dim=1)
        regression = []
        for block,feat in zip(self.regressionModel,features2):
            tmp = block(feat)
            regression.append(tmp)
        regression = torch.cat(regression,dim=1)
        anchors = self.anchors(img_batch)
        #print(classification.shape)
        if self.training:
            #regression = self.regressBoxes(anchors,regression)
            img_shape = img_batch.shape[2:]
            stride = self.anchors.strides
            #anchor_pre_level = cal_num_boxs(img_shape,[4,8,16,32,64])
            return self.loss_head(classification, regression, anchors, annotations)
        else:
            output = self.output(classification,regression,anchors)
            return output

def main():
    net = RetinaNet()

if __name__ == "__main__":
    main()
        
        