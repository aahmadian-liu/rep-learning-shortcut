""" The Automatic Shortcut Removal method (Minderer et al, 2020) """

import numpy as np
import torch
import torch.nn as nn
import time
from algorithms.Algorithm import Algorithm
from pdb import set_trace as breakpoint
import copy
from torch.nn import Softmax
from torch.nn.utils import clip_grad_norm_
import globalconf
import algorithms.Utils as utils


class ShortcutRemoval(Algorithm):

    def __init__(self, opt, run_name="bootadv"):

        super().__init__(opt, run_name)
        self.double_phase = False

        self.classifier = self.networks['classifier']
        self.lens = self.networks['lens']

        self.lamda = opt['lambda']
        self.nclass = opt['n_class']
        self.advloss = opt['adv_loss_type']

        self.totit = 0


    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])

        return self.process_batch(True)

    def evaluation_step(self, batch):
        with torch.no_grad():
            self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
            self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
            return self.process_batch(False)


    def process_batch(self,training):

        record = {}

        self.totit += 1

        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        bsize = dataX.shape[0]

        if training:
            self.optimizers['classifier'].zero_grad()

        y=self.classifier(self.lens(dataX)+dataX)
        y_d=self.classifier(dataX)
        l_c=ce_loss(y,labels)+ce_loss(y_d,labels)
        if training:
            l_c.backward()
            self.optimizers['classifier'].step()


        if training:
            self.optimizers['lens'].zero_grad()

        xp=self.lens(dataX)+dataX
        
        y = self.classifier(xp)

        if self.advloss=='least_likely':
            y_f=torch.argmin(y,dim=1)
            l_g=ce_loss(y,y_f)

        elif self.advloss=='negative':
            l_g=-ce_loss(y,labels)

        rcloss=rloss(dataX, xp)

        l_g= l_g + self.lamda*0.5*rcloss

        if training:
            l_g.backward()
            self.optimizers['lens'].step()


        record['loss_classifier']=[l_c.detach()]
        record['loss_lens']=[l_g.detach()]
        record['loss_l2']=[rcloss.detach()]
        record['acc_classifier']=[accuracy(y, labels, topk=(1,))[0]]

        if training and globalconf.visualize and self.totit%10==0:
            utils.plotimage(xp[0:1],dataX[0:1],'cifar10',self.totit,self.run_name)


        return record


ce_loss=torch.nn.CrossEntropyLoss()

def rloss(r, x):
    return ((r - x) ** 2).mean()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).cpu())
    return res


def to_gpu(x):
    if globalconf.gpu_mode:
        return x.cuda()
    else:
        return x