import cv2
import random
import numpy as np
import math
import os
from dgdet.augment.base import RandomCropObj
from dgdet.augment.base import PadSample
from dgdet.augment.base import HorizontalFlip
from dgdet.augment.base import ResizeSampleSize,ResizeSampleScale
from dgdet.augment.base import FilterSample

class DataAnchorSample(object):
    def __init__(self,anchors=None,aug_ratio=0.6):
        if anchors is None:
            anchors = [12,24,48,96,192,384]
        self.obj_anchors = anchors
        # init shape
        self.shape = [384,640]
        # no aug ratio
        self.defult = 1 - aug_ratio

    def set_img_size(self,shape):
        self.shape = shape
    
    def __call__(self,sample):
        annot = sample['annot']
        # no face in img
        if len(annot)==0:
            sample = ResizeSampleSize(sample,self.shape)
        # infer shape
        elif random.random()<self.defult:
            sample = ResizeSampleSize(sample,self.shape)
        # data anchor sampleing
        else:
            face_idx = random.randint(0,len(annot)-1)
            box = annot[face_idx,:4]
            size = math.sqrt((box[2]-box[0])*(box[3]-box[1]))
            distance = np.array(self.obj_anchors) - size
            distance = np.abs(distance)
            anchor_idx = np.argmin(distance)
            anchor_idx = min(anchor_idx+1,len(self.obj_anchors)-1) #anchor_idx < len(anchors)
            anchor_idx = random.randint(0,anchor_idx)
            # print(anchor_idx)
            anchor = self.obj_anchors[anchor_idx]
            # print(anchor)
            s_target = random.uniform(anchor/2,anchor*2)
            s_target = min(s_target,min(self.shape)-1) #target scale < short side
            scale = s_target/size
            # print(scale)
            sample = ResizeSampleScale(sample,scale)
            sample = RandomCropObj(sample,self.shape,obj_idx=face_idx)
            sample = PadSample(sample,self.shape)
        # filp and filter
        sample = HorizontalFlip(sample,p=0.5)
        sample = FilterSample(sample,min_size=4,max_size=min(self.shape))
        return sample


