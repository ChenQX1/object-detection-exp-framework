import cv2
from dgdet.retinanet.model_tiny import RetinaNet
import torch
from dgdet.transform import resize_sample,to_tensor
import numpy as np
import os
import random
from dgdet.dataset_test import MyDataset

thresh = 0.3

testset = 'testset0915/pair.lst'
fw = open('result/result_tiny.txt',mode='w+')
#os.system('rm -r pred_tiny/*.jpg')

def main():
    #define data
    set1 = MyDataset(testset)

    #define model
    model = RetinaNet(num_classes=1)
    model.eval()
    model.load_state_dict(torch.load('weight/test.pth'))
    model.cuda()

    nb_file = len(set1)
    #count = 0
    for idx in range(nb_file):
        sample = set1[idx]
        ann = sample['annot'].copy()
        name = sample['name']
        print(name)
        img = sample['img'].copy()
        #preprocessing
        shape = [192,112]
        tmp = resize_sample(sample,shape)
        scale = tmp['scale']
        tmp = to_tensor(tmp)
        data = tmp['img']
        data = torch.unsqueeze(data,dim=0)
        data = data.cuda()
        #forward
        result = model(data)
        #print(result)
        scores, classification, transformed_anchors = result
        idxs = np.where(scores.cpu() > thresh)
        
        path = 'testset0915/JPEGImages/{}'.format(name)
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            score = float(scores[j])
            #print(score)
            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            bbox = [x1,y1,x2,y2]
            bbox = list(map(float,bbox))
            #write data
            data = [path]
            data.extend([score])
            data.extend(bbox)
            string = '{} {} {} {} {} {}\n'.format(*data)
            #print(string)
            fw.write(string)
            cv2.putText(img,'{:.3f}'.format(score),(x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.imwrite('pred_tiny/{}.jpg'.format(name),img)
        


if __name__ == "__main__":
    main()

    
