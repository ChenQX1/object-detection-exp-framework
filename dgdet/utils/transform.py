import cv2
import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

# define train config
def resize_sample(sample,shape=[384,640]):
    img = sample['img']
    anns = sample['annot']
    #print(anns.shape)
    boxs = anns[:,:4]
    #label = anns[:,-1]
    train_h,train_w = shape
    back = np.zeros(shape=(train_h,train_w,3),dtype=np.uint8)
    h,w,_ = img.shape
    #cal resize ratio
    scale = 1
    if w>train_w:
        ratio = train_w/w
        w = ratio*w
        h = ratio*h
        scale *=ratio
    if h>train_h:
        ratio = train_h/h
        w = ratio*w
        h = ratio*h
        scale *=ratio
    #do resize
    img = cv2.resize(img,(0,0),fx=scale,fy=scale)
    h_new,w_new,_ = img.shape
    back[:h_new,:w_new,:] = img
    boxs = boxs*scale
    anns[:,:4] = boxs
    sample = {
        'img':back,
        'annot':anns,
        'scale':scale
    }
    return sample


def to_tensor(sample,imagenet=False):
    if imagenet:
        img = sample['img']
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = tf.to_tensor(img)
        img = tf.normalize(img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    else:
        img = tf.to_tensor(sample['img'])
    sample['img'] = img
    sample['annot'] = torch.tensor(sample['annot'])
    return sample

#resize img and pad them together
def collect_train(sample_list):
    #read data
    batch_size = len(sample_list)
    #imgs = [s['img'] for s in sample_list]
    annots = [s['annot'] for s in sample_list]
    #scales = [s['scale'] for s in sample_list]
    max_num_annots = max(annot.shape[0] for annot in annots)
    
    #build empty data
    #train_h,train_w = sample_list[0]['img'].shape[-2:]
    h_list = []
    w_list = []
    for item in sample_list:
        h,w = item['img'].shape[-2:]
        h_list.append(h)
        w_list.append(w)
    train_h = max(h_list)
    train_w = max(w_list)
    padded_imgs = torch.zeros(size=(batch_size,3,train_h,train_w))
    if max_num_annots>0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    # fill empty data
    for idx,sample in enumerate(sample_list):
        h,w = sample['img'].shape[-2:]
        padded_imgs[idx,:,:h,:w] = sample['img']
        n_boxs = sample['annot'].size(0)
        annot_padded[idx,:n_boxs,:] = sample['annot']
    
    #define result
    collected_sample = {
        'img': padded_imgs, 
        'annot': annot_padded, 
    }
    return collected_sample
        
            



