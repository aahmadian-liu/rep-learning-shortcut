""" Downstream classification with the representations obtained from a pretrained model """

import numpy as np
import torch
import torch.nn.functional as functional
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import time

from algorithms.Algorithm import Algorithm
from pdb import set_trace as breakpoint
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
import globalconf

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
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class DownstreamClassifier(Algorithm):
    def __init__(self, opt,run_name='downstream'):
        super().__init__(opt,run_name)
        self.double_phase=False
        
        self.classifier = self.networks['ds_classifier']
        if opt['has_feature_extractor']:
            self.feature_extractor=self.networks['feature_extractor']
        else:
            self.feature_extractor=None
        if opt['has_lens']: #True for ASR method
            self.lens=self.networks['lens']
            self.concat=opt['concat_mode']
        else:
            self.lens=None
        self.flat_features=opt['flat_features'] #if the representation tensor has to be flattened
        self.rep_block=opt['representation_block'] #the block of the feature extractor model to take the representation from

        self.cum_acc=0
        self.cum_acc_num=0

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        
        # Preparing mini-batch of labeled data
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        
        if self.feature_extractor is None: #if a feature extractor model (learned represenation) is not used
            z=dataX
            if self.flat_features:
                z=torch.flatten(z, start_dim=1)
        else:
            if self.lens is None:
                self.feature_extractor.eval()
                with torch.no_grad(): #the feature extractor is frozen
                    z = self.feature_extractor(dataX, rep_block_index=self.rep_block, flatten=self.flat_features) #obtaining the represenation using the pretrained feature extractor model
            else: #when the lens is used in downstream classification
                self.feature_extractor.eval()
                self.lens.eval()
                with torch.no_grad():
                    z1 = self.feature_extractor(self.lens(dataX), rep_block_index=self.rep_block, flatten=self.flat_features)
                    if self.concat: #concatenating the representation with and without lens (ASR method)
                        z2=self.feature_extractor(dataX, rep_block_index=self.rep_block, flatten=self.flat_features)
                        z=torch.cat([z1,z2],dim=1)
                    else:
                        z=z1

         # Computing the loss and accuracy
         
        if do_train:
            self.optimizers['ds_classifier'].zero_grad()

        pred = self.classifier(z)

        record = {}
        loss_total = ce_loss(pred, labels.long())
        record['acc'] = [accuracy(pred.data, labels, topk=(1,))[0]]
        record['loss'] = [loss_total.detach()]

        # Backpropagating and updating the downstream classifier parameters
        if do_train:
            loss_total.backward()
            self.optimizers['ds_classifier'].step()
        
        if not do_train:
            self.cum_acc+=float(record['acc'][0])*float(z.shape[0])
            self.cum_acc_num+=float(z.shape[0])


        return record
        

def togpu(x):
    if globalconf.gpu_mode:
        return x.cuda()
    else:
        return x