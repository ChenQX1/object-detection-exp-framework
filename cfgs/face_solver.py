import os

#-------------------DataSet---------------------
from dgdet.dataset.dataset_face import MyDataset
from cfgs import face_data
Dataset = MyDataset
data_config = face_data

# ------------------Shapes----------------------
train_shapes = [[320,640],[384,640],[448,768]]
infer_shape = [384,640]
#infer_shape = [768,1536]

#-------------------Network--------------------------
'''
Net:nn.Module
'''
from dgdet.detector.retinanet_p3_7 import RetinaNet
Net = RetinaNet
num_class = 1

#-------------------Optim---------------------------
nb_worker = 32
nb_gpus = 8

Epoch = 64
learing_rate = 0.1
#milestones = [32,48]   
warmup_Epoch = 1
wd = 1e-5
batch_size = 256       # batch_size across all gpus
freeze_bn = False

save_pre_iter = 500
imagenet_pretrain = False
pretrain_weight = 'weight/facedet_baseline.pth'
#weight_path = 'weight/facedet_baseline.pth'
weight_path = 'weight/facedet_baseline_pseudo.pth'
