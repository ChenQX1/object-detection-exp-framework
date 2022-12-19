import torch
from torch import nn
from dgdet.src.utils.box import BBoxTransform
from dgdet.src.utils.box import ClipBoxes
from torchvision.ops import nms

class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer,self).__init__()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.use_cuda = True
        self.regressBoxes.use_cuda = self.use_cuda

    def forward(self,classification,regression,anchors, nms_thres):
        transformed_anchors = self.regressBoxes(anchors, regression)
        #transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if self.use_cuda:
            device = classification.get_device()
            finalScores = finalScores.cuda(device)
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda(device)
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda(device)

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > 0.1)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            # anchors_nms_idx = nms(anchorBoxes, scores,0.3)
            anchors_nms_idx = nms(anchorBoxes, scores, nms_thres)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if self.use_cuda:
                device = classification.get_device()
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda(device)

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
        

