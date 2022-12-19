import torch
from torch import nn
from dgdet.src.utils.box import BBoxTransform
from dgdet.src.utils.iou import bbox_overlaps

class GIou_ltrb(nn.Module):
    def __init__(self):
        super(GIou_ltrb,self).__init__()
        #self.loss = IOULoss(loc_loss_type='giou')
        self.transform = BBoxTransform()
    
    def forward(self,anchor,regression,boxs):
        #box transform
        regression = regression.unsqueeze(0)
        anchor = anchor.unsqueeze(0)
        regression = self.transform(anchor,regression) #shape (bs,4)
        regression = regression[0]
        #cal loss
        giou = bbox_overlaps(regression,boxs,mode='giou',is_aligned=True)
        loss = 1-giou #shape = (m,)
        return loss.sum()