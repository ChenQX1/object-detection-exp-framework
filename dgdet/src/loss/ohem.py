import torch
from torch import nn

class OHEM(nn.Module):
    def __init__(self,alpha=3):
        super(OHEM,self).__init__()
        self.alpha = alpha

    def forward(self,classifications,targets):
        classifications = classifications.view(-1)
        targets = targets.view(-1)
        pos_index = (targets==1) #return a list contain(True,False)
        neg_index = (targets==0) #return a list contain(True,False)
        n = pos_index.sum()
        # positive select
        if n>0:
            y = classifications[pos_index]
            loss1 = -torch.log(y)
        else:
            loss1 = torch.tensor([0.]).cuda() 
        # negative select
        n = max(1,n)
        y = classifications[neg_index]
        k = self.alpha*n
        y,_ = torch.topk(y,k,largest=True)
        loss2 = -torch.log(1-y)
        # gether
        loss = torch.cat([loss1,loss2],dim=0)
        loss = loss.sum()
        return loss