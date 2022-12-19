import os
import sys
import torch
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))
import cv2
import time
from torch.utils.data import Dataset,DataLoader
import pseudo_label.cfg as cfg
from pseudo_label.dataset2 import MyDataset
from pseudo_label.model import Model
from pseudo_label.write_xml import gen_txt
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR


#----------init dist-----------
dist.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()  #get gpu id
torch.cuda.set_device(local_rank)

def collate_fn(sample_list):
    return sample_list

def main():
    model = Model(local_rank)
    
    dataset = MyDataset()
    dist_sampler = DistributedSampler(dataset)
    
    dataloader_train = DataLoader(
        dataset = dataset,
        batch_size = 4,
        #shuffle=True,
        num_workers = 2,
        collate_fn = collate_fn,
        sampler = dist_sampler, #if define sampler,do not define shuffle=True
        drop_last=True
        )
    total = len(dataloader_train)
    
    for idx,sample_list in enumerate(dataloader_train):
        if local_rank==0:
            print('local rank: {} process {}/{}'.format(local_rank,idx,total))
        #process batch
        for sample in sample_list:
            img_path = sample['img_path']
            ann_path = sample['ann_path']

            pred = model(sample)
            if len(pred)!=0:
                gen_txt(pred,img_path,ann_path)

if __name__ == '__main__':
    main()





