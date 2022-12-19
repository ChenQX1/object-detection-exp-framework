import math
import torch
from torch import nn
from dgdet.src import head
from dgdet.src import loss
from dgdet.src import assign
from dgdet.src.backbone.darknet import DarkNet
from dgdet.src.neck.fpn import FPN
#from src.neck.fpn_deconv import FPN
from dgdet.src.neck.pan import PAN
from dgdet.src.utils.anchor import Anchors
from dgdet.src.head.repet_conv import Repet1x1
from dgdet.src.head import decode
from dgdet.src.post_process.sigmoid import OutputLayer
from dgdet.src.head.decode import SoftmaxClfHead

class RetinaNet(nn.Module):
    def __init__(self):
        super(RetinaNet,self).__init__()
        #define encode 
        self.backbone = DarkNet(repets=[2,3,3,2],channels=[16,32,48,96,128],dropout=0.1)
        self.neck = FPN([32,48,96,128],cout=64,down=2)
        self.clf_head = nn.ModuleList()
        self.reg_head = nn.ModuleList()
        for _ in range(6):
            self.clf_head.append(Repet1x1(64,2))
            self.reg_head.append(Repet1x1(64,2))
        #define decode 
        self.anchors = Anchors(
            pyramid_levels = [2,3,4,5,6,7],
            sizes = [12,24,48,96,192,384],
            scales=[1],
        )
        self.classificationModel = SoftmaxClfHead(64,num_anchors=1,num_classes=1)
        self.regressionModel = decode.RegHead(64,num_anchors=1)
        self.output = OutputLayer()
        #define loss
        clf_loss = loss.FocalLoss(alpha=0.25,gamma=2.)
        reg_loss = loss.GIou_ltrb()
        st_assign = assign.StanderAssign(pos_thresh=0.4,neg_thresh=0.3)
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
        features = self.neck(features[1:])
        feat_clf = []
        feat_reg = []
        for i,feat in enumerate(features):
            feat = self.clf_head[i](feat)
            feat_clf.append(feat)
            feat = self.reg_head[i](feat)
            feat_reg.append(feat)
        classification = torch.cat([self.classificationModel(feature) for feature in feat_clf], dim=1)
        regression = torch.cat([self.regressionModel(feature) for feature in feat_reg], dim=1)
        anchors = self.anchors(img_batch)
        #print(classification.shape)
        if self.training:
            #regression = self.regressBoxes(anchors,regression)
            img_shape = img_batch.shape[2:]
            stride = self.anchors.strides
            #anchor_pre_level = cal_num_boxs(img_shape,[4,8,16,32,64])
            return self.loss_head(classification, regression, anchors, annotations)
        else:
            classification = torch.sigmoid(classification)
            output = self.output(classification,regression,anchors)
            return output

def main():
    net = RetinaNet()

if __name__ == "__main__":
    main()
        
        