import math
import torch
from torch import nn
import torch.nn.functional as F
from dgdet.src import head
from dgdet.src import loss
from dgdet.src import assign
from dgdet.src.backbone.darknet import DarkNet
from dgdet.src.neck.fpn import FPN
from dgdet.src.neck.pan import PAN
from dgdet.src.utils.anchor import Anchors
from dgdet.src.head import decode
from dgdet.src.post_process.sigmoid import OutputLayer
# new trick
from dgdet.src.loss.margin_focal_loss import FocalLoss
from dgdet.src.loss.marigin_head import LossHead
from dgdet.src.assign.s3fd import S3fdAssign

class RetinaNet(nn.Module):
    def __init__(self,num_class=1):
        super(RetinaNet,self).__init__()
        # deploy
        self.deploying= False
        # define encode 
        self.backbone = DarkNet(repets=[2,3,3,2],channels=[16,32,64,96,128],dropout=0.1)
        self.neck = FPN([64,96,128],cout=64,down=2)
        self.pan = PAN(64,5)
        # define decode 
        self.anchors = Anchors(
            pyramid_levels = [3,4,5,6,7],
            sizes = [16,32,64,128,256],
            scales=[1],
        )
        self.classificationModel = decode.ClfHead(64,num_anchors=1,num_classes=num_class)
        self.regressionModel = decode.RegHead(64,num_anchors=1)
        self.output = OutputLayer()
        # define loss
        #st_assign = assign.StanderAssign(pos_thresh=0.4,neg_thresh=0.3)
        st_assign = S3fdAssign(pos_thresh=0.4,neg_thresh=0.3,min_anchor=3)
        clf_loss = loss.FocalLoss()
        #clf_loss = FocalLoss()
        reg_loss = loss.GIou_ltrb()
        self.loss_head = loss.LossHead(clf_loss,reg_loss,st_assign)
        #self.loss_head = LossHead(clf_loss,reg_loss,st_assign,margin=14)
        self.init_weight()
    
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def deploy(self):
        self.deploying = True
        self.classificationModel.deploying = True
        self.regressionModel.deploying = True

    def init_weight(self):
        #init clf head
        prior = 0.01
        self.classificationModel.conv.weight.data.fill_(0)
        self.classificationModel.conv.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #init reg head
        self.regressionModel.conv.weight.data.fill_(0)
        self.regressionModel.conv.bias.data.fill_(0)

    def forward(self,inputs):
        # train network
        if self.training:
            img_batch, annotations = inputs
        # test network
        else:
            img_batch = inputs
        features = self.backbone(img_batch)
        features = self.neck(features[2:])
        features = self.pan(features)
        # share head
        classification = [self.classificationModel(feature) for feature in features]
        regression = [self.regressionModel(feature) for feature in features]
        if self.deploying:
            #classification = [F.sigmoid(x) for x in classification]
            return classification,regression
        # cat data
        classification = torch.cat(classification, dim=1)
        regression = torch.cat(regression, dim=1)
        # gen anchor
        anchors = self.anchors(img_batch)
        #print(classification.shape)
        if self.training:
            #regression = self.regressBoxes(anchors,regression)
            img_shape = img_batch.shape[2:]
            stride = self.anchors.strides
            #anchor_pre_level = cal_num_boxs(img_shape,[4,8,16,32,64])
            return self.loss_head(classification, regression, anchors, annotations)
        else:
            #clf head shape
            classification = F.sigmoid(classification)
            output = self.output(classification,regression,anchors)
            return output


def main():
    net = RetinaNet()

if __name__ == "__main__":
    main()
        
        
