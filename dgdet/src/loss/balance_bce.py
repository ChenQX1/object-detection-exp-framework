import torch
from torch import nn
import torch.nn.functional as F
import random

class BalanceBCE(nn.Module):
    def __init__(self,ratio=3):
        self.ratio = ratio

    def forward(self,classifications,targets):
        # reshape
        classifications = classifications.view(-1)
        targets = targets.view(-1)
        neg = classifications[targets==0]
        index = range(neg.size(0))
        if targets.sum()==0:
            index = random.choices(index,self.ratio)
            neg = neg[index[:self.ratio]]
            target = 
        # sampling
        loss = F.binary_cross_entropy_with_logits(neg,)
        pos = classifications[targets==1]
        