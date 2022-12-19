import math
import torch
from torch import nn
import torch.nn.functional as F
from dgdet.src import loss
from dgdet.src import assign
from dgdet.src.backbone.resnet import ResNet
from dgdet.src.neck.fpn3x3 import FPN
from dgdet.src.utils.anchor import Anchors
from dgdet.src.head import decode
from dgdet.src.head.repet_dcn import RepetHead
from dgdet.src.post_process.sigmoid import OutputLayer

class RetinaNet(nn.Module):
    def __init__(self):
        super(RetinaNet,self).__init__()
        #define encode 
        self.backbone = ResNet()
        self.neck = FPN([256,512,1024,2048],cout=128,down=1)
        self.clf_head = RepetHead(128,3)
        self.reg_head = RepetHead(128,3)
        #define decode 
        self.anchors = Anchors(
            pyramid_levels = [2, 3, 4, 5, 6],
            sizes = [16,32,64,128,256],
        )
        self.classificationModel = decode.ClfHead1x1(128,num_anchors=1,num_classes=1)
        self.regressionModel = decode.RegHead1x1(128,num_anchors=1)
        self.output = OutputLayer()
        #define loss 
        clf_loss = loss.FocalLoss(alpha=0.25,gamma=2)
        reg_loss = loss.GIou_ltrb()
        st_assign = assign.StanderAssign(pos_thresh=0.5,neg_thresh=0.3)
        self.loss_head = loss.LossHead(clf_loss,reg_loss,st_assign)
        self.init_weight()

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
        features = self.neck(features)
        #head infer
        features_clf = []
        for feat in features:
            features_clf.append(self.clf_head(feat))
        features_reg = []
        for feat in features:
            features_reg.append(self.reg_head(feat))

        classification = torch.cat([self.classificationModel(feature) for feature in features_clf], dim=1)
        regression = torch.cat([self.regressionModel(feature) for feature in features_reg], dim=1)
        anchors = self.anchors(img_batch)
        #print(classification.shape)
        if self.training:
            #regression = self.regressBoxes(anchors,regression)
            img_shape = img_batch.shape[2:]
            stride = self.anchors.strides
            #anchor_pre_level = cal_num_boxs(img_shape,[4,8,16,32,64])
            return self.loss_head(classification, regression, anchors, annotations)
        else:
            #print(anchors.shape)
            classification = F.sigmoid(classification)
            output = self.output(classification,regression,anchors)
            return output

def main():
    net = RetinaNet()

if __name__ == "__main__":
    main()
        
        