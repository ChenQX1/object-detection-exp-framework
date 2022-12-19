import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self,laplace=1e-5,thresh=0.2):
        super(DiceLoss,self).__init__()
        self.laplace = laplace
        self.thresh = thresh

    def forward(self,classifications,targets):
        classifications = classifications.view(-1)
        targets = targets.view(-1)
        clf_neg = classifications[targets==0]
        clf_neg[clf_neg<self.thresh] = 0.
        n = targets.sum()
        if n>0:
            clf_pos = classifications[targets==1].sum()
        else:
            clf_pos = 0
        dice = 1-(2*clf_pos+self.thresh)/(clf_pos+clf_neg.sum()+n+self.thresh)
        return dice

def main():
    a = torch.zeros(200000)
    b = torch.zeros(200000)
    b[0] = 1
    a[0] = 1.
    a[-300:] = 0.2
    model = DiceLoss(laplace=0.1,thresh=0.2)
    loss = model(a,b)
    print(loss)

if __name__ == "__main__":
    main()
        