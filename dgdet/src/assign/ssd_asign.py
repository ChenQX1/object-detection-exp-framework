import torch
from torch import nn
from dgdet.src.utils.iou import bbox_overlaps

'''
positve = 0 ~ nb_gt
negative = -1
ignore = -2
org setting hight=0.5 low=0.4
'''

class StanderAssign(nn.Module):
    def __init__(self,pos_thresh=0.5,neg_thresh=0.3):
        super(StanderAssign,self).__init__()
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
    
    def forward(self,anchor,gt):
        pos_thresh = self.pos_thresh
        neg_thresh = self.neg_thresh

        iou_mat = bbox_overlaps(anchor,gt)
        #assign by anchor
        max_iou,max_iou_index = torch.max(iou_mat,dim=1,keepdim=False)
        #print(max_iou.shape)
        assign_index = torch.ones(anchor.size(0)).long()*-2
        
        #mv tensor to cuda
        if torch.cuda.is_available():
            assign_index = assign_index.cuda()
        
        #pos index1
        pos_index1 = max_iou>pos_thresh
        if pos_index1.sum()!=0:
            assign_index[pos_index1] = max_iou_index[pos_index1]
        
        #neg index
        assign_index[max_iou<neg_thresh] = -1
        #assign by gt
        max_iou,max_iou_index = torch.max(iou_mat,dim=0,keepdim=False)
        #print(max_iou_index)
        for gt_id,anchor_id in enumerate(max_iou_index):
            assign_index[anchor_id] = gt_id

        return assign_index



        

