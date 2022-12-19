import cv2
import torch
import numpy as np
import random
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))
from dgdet.dataset.dataset_test import MyDataset
from dgdet.utils.transform import resize_sample,to_tensor
import cfgs.base_solver as base_solver
from importlib import import_module

#testset = 'facedet_badcase_sideface/image_kunming_bank_imgs_test.lst'
cfg_path = sys.argv[1]
testset = sys.argv[2]
result_path = sys.argv[3]

thresh = 0.2
fw = open(result_path,mode='w+')
os.system('rm -r pred/*.jpg')

def main(cfg=None):
    if cfg is None:
        cfg = base_solver

    #define model
    model = cfg.Net(cfg.num_class)
    model.eval()
    model.load_state_dict(torch.load(cfg.weight_path))
    model.cuda()

    #define data
    set1 = MyDataset(cfg.data_config,testset)
    nb_file = len(set1)
    #count = 0
    for idx in range(nb_file):
        #idx = random.randint(0,nb_file-1)
        sample = set1[idx]
        ann = sample['annot'].copy()
        name = sample['name']
        #print(name)
        img = sample['img'].copy()
        #preprocessing
        shape = cfg.infer_shape
        #shape = [576,960]
        tmp = resize_sample(sample,shape)
        scale = tmp['scale']
        tmp = to_tensor(tmp,cfg.imagenet_pretrain)
        data = tmp['img']
        data = torch.unsqueeze(data,dim=0)
        data = data.cuda()
        #forward
        result = model(data)
        #print(result)
        scores, classification, transformed_anchors = result
        idxs = np.where(scores.cpu() > thresh)
        
        #name = 'facedet_badcase_sideface/image_kunming_bank_imgs_test/{}'.format(name)
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            score = float(scores[j])
            clf = classification[j]
            #print(score)
            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            bbox = [x1,y1,x2,y2]
            bbox = list(map(float,bbox))
            #write data
            data = [name]
            data.extend([score])
            data.extend(bbox)
            string = '{} {} {} {} {} {}\n'.format(*data)
            #print(string)
            fw.write(string)
            cv2.putText(img,'{:.3f}'.format(score),(x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.imwrite('pred/{}.jpg'.format(idx),img)
        


if __name__ == "__main__":
    sovler_path = sys.argv[1]
    solver = sovler_path.split('/')
    solver = '.'.join(solver)[:-3]
    solver = import_module(solver)
    main(solver)

    
