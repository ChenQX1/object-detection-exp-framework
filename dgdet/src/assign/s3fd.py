import torch
from torch import nn
from dgdet.src.utils.iou import bbox_overlaps

'''
positve = 0 ~ nb_gt
negative = -1
ignore = -2
org setting hight=0.5 low=0.4
'''

class S3fdAssign(nn.Module):
    def __init__(self,pos_thresh=0.5,neg_thresh=0.3,min_anchor=3):
        super(S3fdAssign,self).__init__()
        self.use_cuda = True
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.min_anchor = min_anchor
    
    def forward(self,anchor,gt):
        pos_thresh = self.pos_thresh
        neg_thresh = self.neg_thresh

        # init index
        assign_index = torch.ones(anchor.size(0)).long()*-2

        iou_mat = bbox_overlaps(anchor,gt)
        max_iou,max_iou_index = torch.max(iou_mat,dim=1,keepdim=False)
        
        #mv tensor to cuda
        if self.use_cuda:
            assign_index = assign_index.cuda()
        
        # init neg index
        assign_index[max_iou<neg_thresh] = -1

        # assign by thresh
        pos_index1 = max_iou>pos_thresh
        if pos_index1.sum()!=0:
            assign_index[pos_index1] = max_iou_index[pos_index1]
        
        #assign by max and topk
        iou_mat = iou_mat.T # shape = (gt,anchor)
        values,ranks = torch.topk(iou_mat,k=self.min_anchor,dim=1)
        for idx,(v,rank) in enumerate(zip(values,ranks)):
            # assign max boxs
            assign_index[rank[0]] = idx
            # match_anchor<3
            if sum(v>self.pos_thresh)<self.min_anchor:
                index = rank[v>0.1]
                assign_index[index] = idx
        return assign_index



        

