import numpy as np
import random
import os
import cv2
from torch.utils.data import Dataset
from dgdet.utils.utils import read_testset,read_xml
from dgdet.utils import transform
from dgdet.dataset import base_data as data_config
from dgdet.utils.filter import Filter

class MyDataset(Dataset):
    def __init__(self,cfg=None,testset=None):
        if cfg is None:
            cfg = data_config
        self.cfg = cfg
        if testset is None:
            testset = 'facedet_badcase_sideface/image_kunming_bank_imgs_test.lst'
        self.lines = read_testset(cfg.root_dir,testset)
        self.filter = Filter(cfg.tagnames,1)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path = self.lines[idx]['img_path']
        xml_path = self.lines[idx]['ann_path']
        name = img_path[len(self.cfg.root_dir)+1:]
        #read data
        img = cv2.imread(img_path)
        annot = self.filter.read_xml(xml_path)
        sample = {
            'img':img,
            'annot':annot,
            'name':name,
            'img_path':img_path,
            }
        return sample

