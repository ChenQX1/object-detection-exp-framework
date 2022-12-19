import os
import cv2
#from dgdet.dataset.dataset_simple import SimpleDataset
#import cfgs.dangercar_data as cfg
from dgdet.dataset.dataset_face import MyDataset
import cfgs.face_data as cfg

os.system('rm -rf tmp/*.jpg')

def main():
    dataset = MyDataset(cfg,to_tensor=False)
    shape = [320,320]
    for idx in range(10):
        item = [idx,shape]
        sample = dataset[item]
        img = sample['img']
        annot = sample['annot']
        print(img.shape)
        for box in annot:
            x0,y0,x1,y1,id = box
            x0,y0,x1,y1 = list(map(lambda x:int(x),[x0,y0,x1,y1]))
            cv2.rectangle(img, (x0,y0), (x1,y1), color=(0, 0, 255), thickness=2)
        cv2.imwrite('./tmp/{}.jpg'.format(idx),img)

if __name__ == "__main__":
    main()