import random
from dgdet.augment.base import HorizontalFlip,PadSample, RandomCropPatch
from dgdet.augment.base import ResizeSampleScale,ResizeSampleSize
from dgdet.augment.base import FilterSample
from dgdet.augment.base import BaseScale

class ObjAugment(object):
    def __init__(self,aug_ratio=0.6):
        self.aug_ratio = aug_ratio

    def set_img_size(self,shape):
        self.shape = shape

    def __call__(self,sample):
        annot = sample['annot']
        if len(annot)==0:
            sample = ResizeSampleSize(sample,self.shape)
        # no augment
        elif random.random()>self.aug_ratio:
            sample = ResizeSampleSize(sample,self.shape)
        else:
            img = sample['img']
            scale = BaseScale(img,self.shape)
            p = random.uniform(0.6,1.4)
            scale = scale * p
            sample = ResizeSampleScale(sample,scale=scale)
            if p>1:
                sample = RandomCropPatch(sample,max_size=self.shape)
            sample = PadSample(sample,self.shape)
            sample = HorizontalFlip(sample,p=0.5)
        sample = FilterSample(sample,min_size=4,max_size=min(self.shape))
        return sample
            