import numpy as np
import random
import os
import cv2
import torch
from torch.utils.data import Dataset
from dgdet.utils.utils import open_list,read_xml
from dgdet.utils import transform
from dgdet.utils.filter import Filter
from dgdet.augment.data_anchor_sample import DataAnchorSample
from dgdet.dataset import base_data

class MyDataset(Dataset):
    def __init__(self,cfg=None,to_tensor=True):
        if cfg is None:
            cfg = base_data
        self.lines = open_list(cfg.root_dir,cfg.train_list)
        self.to_tensor = to_tensor
        self.imagenet = False
        self.filter = Filter(maps=cfg.tagnames,min_box=5)
        self.aug1 = DataAnchorSample(anchors=[16,32,64,128,256],aug_ratio=0.6)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,item):
        idx,shape = item
        img_path = self.lines[idx]['img_path']
        ann_path = self.lines[idx]['ann_path']
        #read data
        #img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        annot = self.filter.read_xml(ann_path)
        sample = {
            'img':img,
            'annot':annot,
            }
        #augment
        self.aug1.set_img_size(shape)
        sample = self.aug1(sample)
        if self.to_tensor:
            sample = transform.to_tensor(sample,self.imagenet)
        return sample

