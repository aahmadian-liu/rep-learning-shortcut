"""The basic vanilla (empirical risk minimization) method, plus the logits penalty for Spectral Decoupling (Pezeshki et al, 2021)"""

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




class BasicClassification(Algorithm):

    def __init__(self, opt,run_name="bootadv"):

        super().__init__(opt,run_name)

        self.classifier= self.networks['classifier']

        self.nclass = opt['n_class']
        self.logitpenal=opt['logits_penalty']

        self.totit=0
        self.it_in_epoch=0

        
        
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
            self.it_in_epoch=0
            return self.process_batch(False)


    def process_batch(self,training):

        record = {}

        self.totit+=1
        self.it_in_epoch+=1

        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        bsize = dataX.shape[0]

            
        if training:
            self.optimizers['classifier'].zero_grad()

        pred=self.classifier(dataX)

        loss_c= ce_loss(pred,labels)

        if self.logitpenal[0]>0:
            if pred.shape[1]==2:
                predl=pred[:,0]
            else:
                predl=pred
            if self.logitpenal[1]=='mean':
                lsd=(predl**2).mean()
            if self.logitpenal[1]=='sum':
                lsd=(predl**2).sum()

            loss_c+= self.logitpenal[0]*lsd  

        if training:
            loss_c.backward()   
            self.optimizers['classifier'].step()

            
        record['classifier_loss'] = [loss_c.detach()]
        record['classifier_acc']=[accuracy(pred, labels, topk=(1,))[0]]


        return record




softmax=Softmax(dim=1)
ce_loss= torch.nn.CrossEntropyLoss()


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
    

    
 