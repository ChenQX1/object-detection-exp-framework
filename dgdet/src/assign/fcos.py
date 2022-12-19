import torch
from torch import nn
from dgdet.src.utils.distance import box_center,bboxs_distance

'''
implement of center assign
positve = 0 ~ nb_gt
negative = -1
ignore = -2
'''

def cal_edge(boxs):
    w = boxs[:,2] - boxs[:,0]
    h = boxs[:,3] - boxs[:,1]
    edge = (w+h)/2
    return edge

def in_boxs(center,anchor):
    x0 = anchor[:,0].view(-1,1)
    y0 = anchor[:,1].view(-1,1)
    x1 = anchor[:,2].view(-1,1)
    y1 = anchor[:,3].view(-1,1)
    xc = center[0].view(-1,1)
    yc = center[1].view(-1,1)
    result = (xc>x0)*(yc>y0)*(xc<x1)*(yc<y1)
    return result.view(-1)

class FCOS_Assign(nn.Module):
    def __init__(self,scales=[32,64,128,256,512]):
        super(FCOS_Assign,self).__init__()
        # gen scale range
        self.ranges = []
        self.ranges.append([0,scales[0]])
        for i in range(len(scales)-1):
            min = scales[i]
            max = scales[i+1]
            self.ranges.append([min,max])
        # gen shape
        self.shape = []
        for i in range(len(scales)):
            n = 4**i
            self.shape.append(n)
        self.shape.reverse()
        #print(self.shape)

    def forward(self,anchor,gts):
        assign_index = torch.ones(anchor.size(0)).long()*-2
        numb_index = torch.tensor(list(range(anchor.size(0))))
        if torch.cuda.is_available():
            assign_index = assign_index.cuda()
            numb_index = numb_index.cuda()

        # cal anchor
        anchor_min_fpn = int(anchor.size(0)/sum(self.shape))
        anchor_pre_fpn = list(map(lambda x:x*anchor_min_fpn,self.shape))
        #print(anchor_pre_fpn)
        start = 0
        fpn_idx = []
        for i in range(len(anchor_pre_fpn)):
            end = start+anchor_pre_fpn[i]
            fpn_idx.append([start,end])
            start+=anchor_pre_fpn[i]
        
        # center assign
        center_gts = box_center(gts)
        for i,gt in enumerate(gts):
            # loop for each boxs
            edge = torch.sqrt((gt[2]-gt[0])*(gt[3]-gt[1]))
            center = center_gts[i]
            p_level = 0
            for (min,max) in self.ranges:
                if edge>=min and edge<max:
                    break
                else:
                    p_level+=1
            #print(p_level)
            start,end = fpn_idx[p_level]
            anchor_p_level = anchor[start:end]
            position = numb_index[start:end]
            target = in_boxs(center,anchor_p_level)
            # target = torch.zeros(anchor_p_level.size(0))
            # target[0] = 1
            #print(position.shape)
            position = position[target==1]
            assign_index[position] = i
        
        return assign_index



