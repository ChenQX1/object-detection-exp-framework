import os

#-------------------DataSet---------------------
from dgdet.dataset.dataset_simple  import SimpleDataset
# from dgdet.dataset.dataset_headshoulder  import SimpleDataset
from cfgs import headshoulder_data
Dataset = SimpleDataset
data_config = headshoulder_data

# ------------------Shapes----------------------
train_shapes = [[352,640],]
infer_shape = [352,640]


#-------------------Network--------------------------
'''
Net:nn.Module
'''
# from dgdet.detector.base_retinanet import RetinaNet
from dgdet.detector.headshoulder_retinanet import RetinaNet
Net = RetinaNet
num_class = 1
nms_thres = 0.3

#-------------------Optim---------------------------
nb_worker = 8
nb_gpus = 1

Epoch = 64
learing_rate = 0.1
#milestones = [32,48]   
warmup_Epoch = 1
wd = 1e-5
batch_size = 32       # batch_size across all gpus
freeze_bn = False

save_pre_iter = 2000
imagenet_pretrain = False
pretrain_weight = None
weight_path = 'weight/gucci_headshoulder_aug.pth'
