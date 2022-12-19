import numpy as np
import random
import os
import cv2
import torch
from torch.utils.data import Dataset

root_dir = '$HOME/face_det'
data_tag = 'facedet_nonghang'
img_lst = 'img.lst'

class MyDataset(Dataset):
    def __init__(self):
        self.lines = open('{}/{}/{}'.format(root_dir,data_tag,img_lst)).readlines()
        self.lines = list(map(lambda x:'{}/{}'.format(root_dir,x.strip()),self.lines))

    def __len__(self):
        return len(self.lines)

    def random_shape(self,size):
        self.aug.size = size

    def __getitem__(self,idx):
        img_path = self.lines[idx]
        #img_name = img_path.split('/')[-1]
        ann_path = '{}/{}/Annotations/{}.xml'.format(root_dir,data_tag,idx)
        #read data
        #img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        annot = np.zeros(shape=(0,5))
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


