import torch

class WarmUp(object):
    def __init__(self,nb_iter=1,lr=1e-3):
        self._lr_step = lr / nb_iter
        self._running_lr = self._lr_step
        self._end = lr

    def adjust_learning_rate(self,optimizer,lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def step(self,opt):
        if self._running_lr<self._end:
            self.adjust_learning_rate(opt, lr=self._running_lr)
            self._running_lr+=self._lr_step
        else:
            pass
