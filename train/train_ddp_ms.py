import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))
import random
import logging
import collections
import numpy as np
from importlib import import_module
# pytorch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# dgdet
from dgdet.utils.transform import collect_train
from dgdet.utils.sampler import MultiScaleSampler
# cfg
from cfgs import base_solver
# tools
from train.warmup import WarmUp
# ddp pack
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR


#----------init dist-----------
dist.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()  #get gpu id
torch.cuda.set_device(local_rank)
#device = torch.device("cuda", local_rank)


def train(cfg=None):
    if cfg is None:
        cfg = base_solver
    # ---------init logger----------
    model_name = cfg.weight_path.split('/')[-1]
    output_path = './log/{}.log'.format(model_name)
    logging.basicConfig(level=logging.INFO,filename=output_path,filemode='w',format='%(message)s')
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)   #StreamHandler
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)
    
    #------define net work-------
    retinanet = cfg.Net(cfg.num_class)

    #------load weight-----------
    if cfg.pretrain_weight!=None:
        #load weight ddp
        checkpoint = torch.load(cfg.pretrain_weight, map_location=torch.device('cpu'))
        retinanet.load_state_dict(checkpoint,strict=False)

    retinanet.cuda(local_rank)
    
    if cfg.freeze_bn:
        retinanet.module.freeze_bn()
    retinanet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(retinanet)
    retinanet = DistributedDataParallel(retinanet,device_ids=[local_rank])

    opt = optim.SGD(
        retinanet.parameters(),
        lr=cfg.learing_rate,
        weight_decay=cfg.wd,
        momentum=0.9)

    #-------define dataset-----------
    dataset_train = cfg.Dataset(cfg.data_config,to_tensor=True)
    dataset_train.imagenet = cfg.imagenet_pretrain
    
    #-------define sampler-----------
    dist_sampler = DistributedSampler(dataset_train)
    ms_sampler = MultiScaleSampler(
        dist_sampler,
        batch_size = cfg.batch_size//cfg.nb_gpus,
        drop_last = True,
        multiscale_step = 1,
        img_sizes = cfg.train_shapes,
        )

    dataloader_train = DataLoader(
        dataset = dataset_train,
        #batch_size = cfg.batch_size//cfg.nb_gpus,
        #shuffle=True,
        num_workers = cfg.nb_worker//cfg.nb_gpus,
        collate_fn = collect_train,
        #sampler=train_sampler, #if define sampler,do not define shuffle=True
        batch_sampler=ms_sampler,
        )

    #-------define warmup------------
    nb_iter = len(dataloader_train)*cfg.warmup_Epoch
    warmup = WarmUp(nb_iter=nb_iter,lr=cfg.learing_rate)

    #-------define lr_scheduler------
    lr_scheduler = CosineAnnealingLR(opt,T_max=(cfg.Epoch-cfg.warmup_Epoch))
    #lr_scheduler = MultiStepLR(opt,milestones=cfg.milestones,gamma=0.1)

    #--------train loop-------------
    for epoch_num in range(cfg.Epoch):
        total_loss_clf = 0
        total_loss_reg = 0
        dist_sampler.set_epoch(epoch_num) #set random dataloader
        if local_rank==0:
            logger.info('Epoch:{}'.format(epoch_num))
            if epoch_num>0:
                lr_scheduler.step() #update lr 
                #save weight
                torch.save(retinanet.module.state_dict(),cfg.weight_path) #mutil gpus
        for iter_num, data in enumerate(dataloader_train):
            #warm up
            warmup.step(opt)
            opt.zero_grad()

            #geather mutil gpus output
            classification_loss,regression_loss,n = retinanet([data['img'].cuda().float(), data['annot']])
            classification_loss = classification_loss.sum()
            regression_loss = regression_loss.sum()
            n = n.sum()
            if n>0:
                classification_loss = classification_loss/n
                regression_loss = regression_loss/n
            # classification_loss = classification_loss.mean()
            # regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            
            #----update parametrs----------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            opt.step()

            #loop hook
            lr = opt.param_groups[0]['lr'] 
            if iter_num%10==0 and local_rank==0:
                h,w = data['img'].shape[-2:]
                string='Epoch:{} iter:{} {}x{} lr:{:.3f} clf:{:1.5f} reg:{:1.5f} sum:{:1.5f}'.format(
                        epoch_num, iter_num,
                        h,w,lr,
                        float(classification_loss), float(regression_loss), float(loss))
                logger.info(string)
            #save weight pre gpus
            if iter_num%cfg.save_pre_iter==0 and iter_num!=0 and local_rank==0:
                torch.save(retinanet.module.state_dict(),cfg.weight_path) #mutil gpus

            #cal total loss
            total_loss_clf += float(classification_loss)
            total_loss_reg += float(regression_loss)

        #cal mean loss
        total_loss_clf /= iter_num
        total_loss_reg /= iter_num
        sum = total_loss_clf + total_loss_reg
        if local_rank==0:
            string = 'Epoch:{} clf: {:1.5f} reg: {:1.5f} sum: {:1.5f}'.format(
                            epoch_num,total_loss_clf, total_loss_reg,sum) 
            logger.info(string)

if __name__ == '__main__':
    sovler_path = sys.argv[-1]
    solver = sovler_path.split('/')
    solver = '.'.join(solver)[:-3]
    solver = import_module(solver)
    train(solver)
