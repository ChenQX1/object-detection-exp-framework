import os
from pseudo_label.dataset2 import MyDataset
from pseudo_label.model import Model
import cv2
import numpy as np

os.system('rm -rf tmp/*.jpg')

def main():
    set1 = MyDataset()
    model = Model(device=1)
    idx_list = list(range(0,10,1))
    for idx in idx_list:
        sample = set1[idx]
        print(sample['img_path'])
        #print(sample['ann_path'])
        img = sample['img']
        # vis org
        org = img.copy()
        for box in sample['annot'][:,:4]:
            x0,y0,x1,y1 = list(map(int,box))
            cv2.rectangle(org,(x0,y0),(x1,y1), color=(0,255,0), thickness=2)
        #cv2.imwrite('tmp/org.jpg',org)
        # vis pred
        boxes = model(sample)
        for box in boxes:
            x0,y0,x1,y1 = box
            cv2.rectangle(img,(x0,y0),(x1,y1), color=(0, 0, 255), thickness=2)
        #img = np.concatenate([org,img],axis=1)
        cv2.imwrite('tmp/model_{}.jpg'.format(idx),img)

if __name__ == "__main__":
    main()