import numpy as np
import cv2
from torch.utils.data import Dataset
from dgdet.utils import transform
from dgdet.utils.utils import open_list
from dgdet.utils.filter import Filter
from dgdet.dataset import base_data
from dgdet.augment.obj_aug import ObjAugment

class SimpleDataset(Dataset):
    def __init__(self,cfg=None,to_tensor=True):
        if cfg is None:
            cfg = base_data
        self.lines = open_list(cfg.root_dir,cfg.train_list)
        self.to_tensor = to_tensor
        self.imagenet = False
        self.filter = Filter(maps=cfg.tagnames,min_box=5)
        self.augment = ObjAugment(aug_ratio=0.6)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,item):
        idx,shape = item
        img_path = self.lines[idx]['img_path']
        ann_path = self.lines[idx]['ann_path']
        # read data
        #img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        annot = self.filter.read_xml(ann_path)
        sample = {
            'img':img,
            'annot':annot
            }
        # resize
        if shape!='org' and isinstance(shape,list):
            #sample = transform.resize_sample(sample,shape)
            #print(sample.keys())
            self.augment.set_img_size(shape)
            sample = self.augment(sample)
        # to_tensor
        if self.to_tensor:
            sample = transform.to_tensor(sample,self.imagenet)
        return sample

