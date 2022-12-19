import os

#-------------------DataSet---------------------
from dgdet.dataset.dataset_simple import SimpleDataset
from cfgs import dangercar_data
Dataset = SimpleDataset
data_config = dangercar_data

#-------------------shapes----------------------
train_shapes = [[320,320],]
infer_shape = [320,320]

#-------------------Network---------------------
'''
Net:nn.Module
'''
from dgdet.detector.base_retinanet import RetinaNet
Net = RetinaNet
num_class = 2

#-------------------Optim-----------------------
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
pretrain_weight = None
weight_path = 'weight/danger_car.pth'
