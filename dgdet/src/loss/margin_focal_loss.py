import torch
from torch import nn
import torch.nn.functional as F

'''
target:
1  : positive
0  : negative
-1 : ignore
'''

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,classifications,targets,marign):
        alpha = self.alpha
        gamma = self.gamma
        marign = marign.view(-1)
        classifications = classifications.view(-1)
        targets = targets.view(-1)
        
        #reduce margin
        if targets.sum()>0:
            classifications[targets==1] -= marign[targets==1]
        
        #focal loss
        p = torch.sigmoid(classifications)
        ce_loss = F.binary_cross_entropy_with_logits(classifications, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.sum()