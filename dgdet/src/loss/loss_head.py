import torch
from torch import nn

class LossHead(nn.Module):
    def __init__(self,clf_loss,reg_loss,assign):
        super(LossHead,self).__init__()
        self.clf_loss = clf_loss
        self.reg_loss = reg_loss
        self.anchor_asign = assign

    def forward(self,classifications,regressions,anchors,annotations):
        # collect data
        classification_losses = []
        regression_losses = []
        pos = []
        # cal loss for each data
        batch_size = classifications.shape[0]
        anchor = anchors[0, :, :]
        for j in range(batch_size):
            #obtain head from batch
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1] #delete empty bbox
            # img without bbox
            if bbox_annotation.shape[0] == 0:
                clf_target = torch.zeros([classification.size(0)]).cuda().long()
                # cal loss
                clf_loss = self.clf_loss(classification,clf_target)
                # collect loss
                classification_losses.append(clf_loss)
                regression_losses.append(torch.tensor(0).float().cuda())
                n = torch.tensor([0]).cuda().float()
                pos.append(n)
                continue
            #p:0~n,n:-1,ignore:-2
            assign_index = self.anchor_asign(anchor,bbox_annotation[:,:4])
            n = (assign_index>=0).sum().float().view(-1)
            annot_index = assign_index[assign_index>=0]
            #print(assign_index.shape)
            #print(classification.shape)
            # classification loss
            target = bbox_annotation[:,4].long().view(-1) # Anchor x 1
            target = target[annot_index]
            clf_target = torch.zeros([classification.size(0)]).cuda().long() # Anchor x 1
            clf_target[assign_index>=0] = target      # positive
            clf_target[assign_index==-1] = 0          # negative
            clf_target[assign_index==-2] = -1         # ignore
            # drop ignore
            classification = classification[clf_target!=-1]
            clf_target = clf_target[clf_target!=-1]
            # cal clf loss
            clf_loss = self.clf_loss(classification,clf_target)
            # regression loss 
            anchor_r = anchor[assign_index>=0]
            regression = regression[assign_index>=0]
            #annot_index = assign_index[assign_index>=0]
            annot = bbox_annotation[annot_index]
            boxs = annot[:,:4]
            reg_loss = self.reg_loss(anchor_r,regression,boxs)
            # collect loss
            classification_losses.append(clf_loss)
            regression_losses.append(reg_loss)
            pos.append(n)
        # stack loss
        classification_losses = torch.stack(classification_losses) 
        regression_losses = torch.stack(regression_losses)
        n = torch.stack(pos)
        return classification_losses,regression_losses,n

