import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.0):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,classifications,targets):
        alpha = self.alpha
        gamma = self.gamma
        classifications = classifications.view(-1)
        targets = targets.view(-1)
        #focal loss
        p = torch.sigmoid(classifications)
        ce_loss = F.binary_cross_entropy_with_logits(classifications, targets.float(), reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.sum()

class MultiClassFocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.):
        super(MultiClassFocalLoss,self).__init__()
        self.loss = FocalLoss(alpha,gamma)
    
    def forward(self,classifications,targets):
        loss = []
        targets = targets.view(-1)
        # embedding onehot label
        nb_class = classifications.size(1)
        targets = F.one_hot(targets,nb_class+1) # Anchor x num_class+1 
        targets = targets[:,1:] # delate background
        for i in range(nb_class):
            loss_tmp = self.loss(classifications[:,i],targets[:,i])
            loss.append(loss_tmp)
        loss = torch.stack(loss).mean()
        return loss
            
            


