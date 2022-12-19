import numpy as np
import random
import os
import cv2
import torch
from torch.utils.data import Dataset
from dgdet.utils.utils import open_list,read_xml


class MyDataset(Dataset):
    def __init__(self):
        self.lines = open_list()

    def __len__(self):
        return len(self.lines)

    def random_shape(self,size):
        self.aug.size = size

    def __getitem__(self,idx):
        img_path = self.lines[idx]['img_path']
        ann_path = self.lines[idx]['ann_path']
        #read data
        #img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        boxs = read_xml(ann_path)
        nb_boxs = len(boxs)
        if nb_boxs>0:
            label = np.zeros(shape=(nb_boxs,1)) # 0 = sigmoid 0 
            boxs = np.array(boxs)
            annot = np.concatenate((boxs,label),axis=1)
        else:
            annot = np.ones(shape=(0,5),dtype=np.float32)
        sample = {
            'img':img,
            'annot':annot,
            'img_path':img_path,
            'ann_path':ann_path,
            }
        #augment
        return sample


def main():
    set1 = MyDataset()
    sample = set1[0]
    img = sample['img']
    cv2.imwrite('tmp/img.jpg',img)

if __name__ == "__main__":
    main()


