""" The Diverse Ensemble method (Teney et al, 2022) """

import numpy as np
import torch
import torch.nn as nn
import time
from algorithms.Algorithm import Algorithm
from pdb import set_trace as breakpoint
from torch.nn import Softmax
from torch.autograd import grad
import globalconf


class DiverseEnsemble(Algorithm):

    def __init__(self, opt, run_name="diverseens"):

        super().__init__(opt, run_name)
        self.double_phase = False

        self.ensemble = self.networks['ensemble']

        self.lambda_div = opt['lambda_div']

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
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        return self.process_batch(False)

    def process_batch(self,training):

        start = time.time()

        record = {}

        self.totit += 1

        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        bsize = dataX.shape[0]

        if training:
            self.optimizers['ensemble'].zero_grad()

        y,h=self.ensemble(dataX,return_h=True)
        y=to_gpu(y)
        h=to_gpu(h)

        l1=to_gpu(torch.zeros(1))

        nc=self.ensemble.n_class
        nens=self.ensemble.n_classifiers

        gmat=to_gpu(torch.zeros(bsize,nens,h.shape[1]))

        for m in range(nens):

            y_c=y[:,m*nc:(m+1)*nc]

            l1+= ce_loss(y_c,labels)

            y_cm=torch.gather(y_c,1,torch.argmax(y_c,dim=1,keepdim=True))

            y_cm=y_cm.sum()
            
            gr=grad(y_cm,h,create_graph=True)[0]
            gr=gr*1.0/(torch.sqrt(torch.sum(gr**2,dim=1,keepdim=True))+0.0001)
            gmat[:,m,:]=gr

        l1=l1/nens

        gmat_t=torch.transpose(gmat,1,2)

        s = torch.matmul(gmat,gmat_t)

        l2= (s.sum()-torch.diagonal(s,0,1,2).sum()) /bsize
        l2=l2/(nens*(nens-1))

        l_t = l1 + self.lambda_div*l2

        if training:
            l_t.backward()
            self.optimizers['ensemble'].step()

        record['loss']=[l_t.detach()]
        record['loss_c']=[l1.detach()]
        record['loss_d'] = [l2.detach()]

        acmean=0

        for m in range(nens):

            y_c=y[:,m*nc:(m+1)*nc]

            acmean+=accuracy(y_c,labels)[0]
        
        acmean=acmean/nens

        record['acc_mean']=[acmean]


        return record


softmax = Softmax(dim=1)
ce_loss = torch.nn.CrossEntropyLoss()


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