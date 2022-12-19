import torch
from torch import nn
from dgdet.src.utils.iou import bbox_overlaps
from torchvision.ops import nms

'''
positve = 0 ~ nb_gt
negative = -1
ignore = -2
org setting hight=0.5 low=0.4
'''

class AssignYolo(nn.Module):
    def __init__(self,thresh=0.3):
        super(AssignYolo,self).__init__()
        self.thresh = thresh
    
    def forward(self,anchor,gt):
        thresh = self.thresh
        assign_index = torch.ones(anchor.size(0)).long()*-2
        #mv tensor to cuda
        if torch.cuda.is_available():
            assign_index = assign_index.cuda()
        #cal iou
        iou_mat = bbox_overlaps(anchor,gt)
        max_iou,max_iou_index = torch.max(iou_mat,dim=1,keepdim=False)
        #asign negative
        assign_index[max_iou<thresh] = -1
        #assign positve
        max_iou,max_iou_index = torch.max(iou_mat,dim=0,keepdim=False)
        #print(max_iou_index)
        for gt_id,anchor_id in enumerate(max_iou_index):
            assign_index[anchor_id] = gt_id

        return assign_index