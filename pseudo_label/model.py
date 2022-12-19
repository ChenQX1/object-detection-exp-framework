import torch
import cv2
import numpy as np
from torchvision.transforms import functional as tf
from torchvision.ops import nms
from torch.nn.parallel.distributed import DistributedDataParallel
import pseudo_label.cfg as cfg


class Model(object):
    def __init__(self,device=0):
        self.device = device
        self.model = cfg.Net()
        weight = torch.load(cfg.weight_path,map_location=torch.device('cpu'))
        self.model.load_state_dict(weight)
        self.model.cuda(self.device)
        self.model.eval()
        self.thresh = cfg.thresh
        self.h,self.w = cfg.infer_shape
        self.step = cfg.step
        self.scale = cfg.scale

    def to_tesnor(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = tf.to_tensor(img)
        img = tf.normalize(img,cfg.mean,cfg.std)
        img = img.unsqueeze(0)
        #img = img.cuda()
        return img
    
    def forward(self,data):
        result = self.model(data)
        scores, classification, transformed_anchors = result
        idxs = np.where(scores.cpu() > self.thresh)
        annot = []
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x0,y0,x1,y1 = list(map(float,bbox))
            score = float(scores[j])
            pred = [x0,y0,x1,y1,score]
            annot.append(pred)
        annot = np.array(annot)
        return annot

    def make_batch(self,img):
        h,w,_ = img.shape
        if h<self.h and w<self.w:
            back = np.zeros(shape=(self.h,self.w,3),dtype=np.uint8)
            back[0:h,0:w,:] = img
            data = self.to_tesnor(back)
            data = data.cuda()
            coords = [[0,0]]
            return data,coords
        data = []
        coords = []
        for start_y in range(0,h,self.step):
            end_y = start_y + self.h
            if end_y>h:
                end_y = h
            for start_x in range(0,w,self.step):
                #print(start_y,start_x)
                end_x = start_x + self.w
                if end_x>w:
                    end_x = w
                tmp = img[start_y:end_y,start_x:end_x,:]
                #print(tmp.shape)
                tmp_h,tmp_w = tmp.shape[:2]
                #pad tmp while size is not fit
                if tmp_w<self.w or tmp_h<self.h:
                    back = np.zeros(shape=(self.h,self.w,3),dtype=np.uint8)
                    back[0:tmp_h,0:tmp_w,:] = tmp
                    tmp = back
                #print(tmp.shape)
                #mv tensor
                tmp = self.to_tesnor(tmp)
                data.append(tmp)
                coords.append([start_x,start_y])
        data = torch.cat(data,dim=0)
        #print(data.shape)
        return data,coords

    def _infer_img(self,img):
        # make batch
        batch_data,coords = self.make_batch(img)
        batch_data = batch_data.cuda(self.device)
        collect = []
        # infer batch
        for data,(x,y) in zip(batch_data,coords):
            data = data.unsqueeze(0)
            pred = self.forward(data)
            if len(pred)>0:
                pred = pred + np.array([x,y,x,y,0])
                collect.append(pred)
        if len(collect)>0:
            collect = np.concatenate(collect,axis=0)
        return collect

    def __call__(self,sample):
        img = sample['img']
        annot = sample['annot']
        collect = np.zeros(shape=(0,5))
        #multi scale infer
        for s in self.scale:
            tmp = cv2.resize(img,(0,0),fx=s,fy=s)
            #print(tmp.shape)
            dets = self._infer_img(tmp)
            if len(dets)>0:
                dets = np.array(dets)
                #print(dets.shape)   
                dets[:,:4] = dets[:,:4]/s
                collect = np.concatenate([collect,dets],axis=0)
        # no face find
        if len(collect)==0:
            return annot[:,:4]
        # mearge with annot
        if len(annot)>0:
            tmp = np.ones(shape=(annot.shape[0],5))
            tmp[:,:4] = annot[:,:4]
            collect = np.concatenate([collect,tmp],axis=0)
        # mv collect
        collect = torch.tensor(collect)
        # do nms
        box = collect[:,:4]
        score = collect[:,-1]
        keep_index = nms(box,score,iou_threshold=0.1)
        box = box[keep_index]
        # get result
        box = box.numpy().astype(np.int).tolist()
        return box

        