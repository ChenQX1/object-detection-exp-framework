import cv2
import numpy as np
import random
import os

def HorizontalFlip(sample,p=1.):
    if random.random()>p:
        return sample
    img = sample['img']
    annot = sample['annot']
    #flip box
    h,w,_ = img.shape
    box = annot[:,:4].copy()
    x_max = w - box[:, 0]
    x_min = w - box[:, 2]
    box[:, 0] = x_min
    box[:, 2] = x_max
    annot[:,:4] = box
    #flip img
    img = cv2.flip(img,1)
    sample = {
        'img':img,
        'annot':annot
    }
    return sample

def VerticalFlip(sample,p):
    if random.random()>p:
        return sample
    img = sample['img']
    img = cv2.flip(img,flipCode=0)
    annot = np.zeros(shape=(0,5))
    sample = {
        'img':img,
        'annot':annot
    }
    return sample

def BaseScale(img,target_size):
    h,w,_ = img.shape
    train_h,train_w = target_size
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
    return scale

def RandomScale(sample,scale=[1.],p=1.):
    if random.random()>p:
        return sample
    img = sample['img']
    annot = sample['annot']
    #cal random scale
    scale = random.choice(scale)
    #print(base_scale,scale)
    h,w,_ = img.shape
    if min(scale*h,scale*w)<20:
        scale = 20/min(w,h)
    img = cv2.resize(img,(0,0),fx=scale,fy=scale)
    annot[:,:4] = annot[:,:4]*scale
    sample = {
        'img':img,
        'annot':annot
    }
    return sample


def boxs_overlap(a,b):
    area1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    l = np.maximum(a[:,0],b[:,0])
    r = np.minimum(a[:,2],b[:,2])
    t = np.maximum(a[:,1],b[:,1])
    b = np.minimum(a[:,3],b[:,3])

    over_lap = (r-l)*(b-t)
    mask = (r>l)*(b>t)
    over_lap = over_lap*mask

    IoU = over_lap / (area1+1e-4)

    return IoU

def delate_boxs(boxs,roi):
    roi_l,roi_t,roi_r,roi_b = roi
    xc = (boxs[:,0] + boxs[:,2])/2
    yc = (boxs[:,1] + boxs[:,3])/2
    keep = (xc>roi_l)*(xc<roi_r)*(yc>roi_t)*(yc<roi_b)
    return keep

'''
if roi > img shape roi while be clip to img shape
roi shape must be (1,4)
'''
def CropSample(sample,roi):
    img = sample['img']
    annot = sample['annot']
    x0,y0,x1,y1 = roi[0]
    h,w,_ = img.shape
    #clip roi in img shape
    x0 = max(x0,0)
    y0 = max(y0,0)
    x1 = min(x1,w)
    y1 = min(y1,h)
    img = img[y0:y1,x0:x1]
    #clip boxs
    overlaps = boxs_overlap(annot[:,:4],roi).reshape(-1) # shape=(n,)
    #print(overlaps)
    keep_index = overlaps>0.3
    if keep_index.sum()==0:
        annot = -np.ones(shape=(0,5))
    else:
        annot = annot[keep_index,:]
        annot[:,:4] = annot[:,:4] - np.array([x0,y0,x0,y0])
    #print(annot)
    sample = {
        'img':img,
        'annot':annot
    }
    return sample
    
def RandomCropPatch(sample,max_size=[384,640],p=1.):
    patch_h,patch_w = max_size
    if random.random()>p:
        return sample
    img = sample['img']
    h,w,_ = img.shape
    if h<patch_h and w<patch_w:
        return sample
    #random roi
    x0 = int(random.uniform(0,w-patch_w))
    y0 = int(random.uniform(0,h-patch_h))
    x1 = x0 + patch_w
    y1 = y0 + patch_h
    roi = np.array([x0,y0,x1,y1]).reshape(-1,4)
    #crop sample
    sample = CropSample(sample,roi)
    return sample

def RandomCropObj(sample,size=[384,640],obj_idx=1):
    ht,wt = size
    img = sample['img']
    h,w,_ = img.shape
    if h<ht and w<wt:
        return sample
    else:
        face = sample['annot'][obj_idx]
        xc = (face[0]+face[2])/2
        yc = (face[1]+face[3])/2
        x0 = xc + random.uniform(-0.2,0.2)*wt - wt/2
        y0 = yc + random.uniform(-0.2,0.2)*ht - ht/2
        x0 = int(max(0,x0))
        y0 = int(max(0,y0))
        x1 = x0 + wt
        y1 = y0 + ht
        roi = np.array([x0,y0,x1,y1]).reshape(-1,4)
        #crop sample
        sample = CropSample(sample,roi)
    return sample

def FilterSample(sample,min_size=4,max_size=144):
    img = sample['img']
    h,w,_ = img.shape
    annot = sample['annot']
    if annot[:,-1].sum()==0:
        return sample
    annot_pad = np.ones(shape=(0,5))*-1
    for idx in range(annot.shape[0]):
        x0,y0,x1,y1,_ = annot[idx]
        if max((x1-x0),(y1-y0))>min_size or max((x1-x0),(y1-y0))<max_size:
            tmp = annot[idx].reshape(1,-1)
            annot_pad = np.concatenate((annot_pad,tmp),axis=0)
    sample['img'] = img
    sample['annot'] = annot_pad
    return sample


def SamePadSample(sample,shape):
    pad_h,pad_w = shape
    img = sample['img']
    annot = sample['annot']
    h,w,_ = img.shape
    img1 = np.zeros((pad_h,pad_w,3),dtype=np.uint8)
    annot1 = np.zeros(shape=(0,5))
    y0 = 0
    while(y0<pad_h):
        y1 = min(y0+h,pad_h)
        x0 = 0
        while(x0<pad_w):
            x1 = min(x0+w,pad_w)
            crop_w = x1-x0
            crop_h = y1-y0
            patch = img[:crop_h,:crop_w,:]
            #photo aug for patch y=w*x+b
            #patch = RandomContrast(patch.copy(),p=0.7)
            #patch = RandomBrightness(patch.copy(),p=0.7)
            #copy img 
            img1[y0:y1,x0:x1,:] = patch
            tmp = annot.copy()
            tmp[:,:4] = tmp[:,:4] + np.array([x0,y0,x0,y0])
            annot1 = np.concatenate([annot1,tmp],axis=0)
            x0 = x1
        y0 = y1
    if annot1.shape[0]>0:
        keep = delate_boxs(annot1[:,:4],[0,0,pad_w,pad_h])
        annot1 = annot1[keep]
    sample = {
        'img':img1,
        'annot':annot1
    }
    return sample

def ResizeSampleScale(sample,scale):
    img = sample['img']
    annot = sample['annot']
    img = cv2.resize(img,(0,0),fx=scale,fy=scale)
    annot[:,:4] = annot[:,:4]*scale
    sample = {
        'img':img,
        'annot':annot
    }
    return sample

def ResizeSampleSize(sample,shape=[384,640]):
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
    h,w,_ = img.shape
    back[:h,:w,:] = img
    anns[:,:4] = anns[:,:4]*scale
    sample = {
        'img':back,
        'annot':anns,
    }
    return sample

def PadSample(sample,size):
    h,w = size
    back = np.zeros(shape=(h,w,3),dtype=np.uint8)
    h,w = sample['img'].shape[:2]
    back[:h,:w,:] = sample['img']
    sample['img'] = back
    return sample

def RandomTranslate(sample):
    pass



