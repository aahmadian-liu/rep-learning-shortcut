""" The proposed Adversarial Lens with Random transform (ALR) method """


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




class AdvLensRan(Algorithm):

    def __init__(self, opt,run_name="alr"):

        super().__init__(opt,run_name)
        self.double_phase = False

        self.classifier= self.networks['classifier']
        self.lens= self.networks['lens']

        self.warmup_iters=opt['warmup_iterations']
        self.beta = opt['beta']
        self.nclass = opt['n_class']
        self.steps_c=opt['steps_classifier']
        self.steps_l = opt['steps_lens']
        self.vanilmode= opt['vanilla_classifier_mode']
        self.grad_max=opt['grad_clip_max']
        self.lensloss=opt['lens_loss_config']
        imagesize=opt['image_size']

        self.totit=0
        self.it_in_epoch=0

        self.randmat=to_gpu(torch.rand(imagesize[1],imagesize[1])) # the random transform matrix

        
        
    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])

        return self.process_batch_train()


    def evaluation_step(self, batch):
        with torch.no_grad():
            self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
            self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
            self.it_in_epoch=0
            return self.process_batch_test()


    def process_batch_train(self):

        record = {} #training statistics on current minibatch

        self.totit+=1
        self.it_in_epoch+=1

        no_advers = self.vanilmode or (self.totit <= self.warmup_iters) #if adversarial training part should be skipped

        if self.vanilmode:
            print("basic classifier mode")

        # input data minibatch
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        bsize = dataX.shape[0]


        # Applying the random transform (multiplying input images by R)
        if not self.vanilmode:
            Xp=to_gpu(torch.zeros_like(dataX))
            for k in range(3):
                cp=torch.matmul(dataX[:,k,:,:],self.randmat)
                Xp[:,k,:,:]=cp

        # Computing the lens loss, and updating the lens parameters

        if not no_advers:
            # Switching the training/eval modes of the networks
            self.classifier.eval()
            self.lens.train()

            for i in range(self.steps_l):

                self.optimizers['lens'].zero_grad()

                # Reconstructing the input images by the lens, and obtaining the error based on either feature matching or cross-entropy
                xfk=self.lens(Xp)

                if self.lensloss[0]=='features':
                    z = self.classifier(dataX,rep_block_index=self.lensloss[2] )
                    zf= self.classifier(xfk,rep_block_index=self.lensloss[2])

                    if self.lensloss[1]=='norm1':
                        loss_g = torch.abs(z - zf).mean()
                    if self.lensloss[1]=='norm2':
                        loss_g = rloss(z,zf)

                if self.lensloss[0] == 'ce':
                    predf=self.classifier(xfk)
                    loss_g = ce_loss(predf, labels)

                # backpropagation and taking gradient step
                loss_g.backward()
                self.optimizers['lens'].step()

                record['lens_loss'] = [loss_g.detach()]

                print("lens step")
        else:
            record['lens_loss'] = [to_gpu(torch.tensor(0.0))]

        
        # Switching the training/eval modes of the networks
        self.classifier.train()
        self.lens.eval()

        # Obtaining the fake reconstructions and labeling them
        if not self.vanilmode and self.beta>0:
            yfk= to_gpu(torch.ones_like(labels,dtype=torch.long))
            yfk[:]=self.nclass
            xfk= self.lens(Xp)

        # Computing the classifier loss, and updating the classsifier parameters

        for i in range(self.steps_c):
            
            self.optimizers['classifier'].zero_grad()

            # Merging the real and fake samples
            if not self.vanilmode and self.beta>0:
                xj=torch.cat([dataX,xfk])
            else:
                xj = dataX
                print('no fake samples')

            predj=self.classifier(xj)
        
            pred1=predj[0:bsize,:] # classifier output on the real samples

            l1=ce_loss(pred1,labels) # classification loss
            

            if not no_advers and self.beta>0:
                pred2=predj[bsize:,:] # classifier output on the fake samples
                l2=ce_loss(pred2,yfk) # adverrsarial discrimination loss
            else:
                l2=to_gpu(torch.tensor(0.0))
                print('no adverse loss')
            

            loss_c= l1 + self.beta*l2  # total classifier loss

            # backpropagation and taking gradient step
            loss_c.backward()
            if self.grad_max>0:
                clip_grad_norm_(self.classifier.parameters(),self.grad_max)
            self.optimizers['classifier'].step()

            print("classifier step")

            # Saving the loss and accuracy values
            if i==self.steps_c-1 :
                record['classifier_adv_loss'] = [l2.detach()]
                record['classifier_acc']=[accuracy(pred1, labels, topk=(1,))[0]]
                record['classifier_acc_trun'] = [accuracy(pred1[:,0:self.nclass], labels, topk=(1,))[0]] #accuracy without the last extra neuron
                record['classifier_ce_loss'] = [l1.detach()]
                if not no_advers and self.beta>0:
                    record['classifier_recal_fake'] = [accuracy(pred2, yfk, topk=(1,))[0]]
                else:
                    record['classifier_recal_fake']=[torch.tensor(0.0)]
                    record['classifier_loss']=[loss_c.detach()] 

        # Plotting the images and reconstructions if enabled
        if globalconf.visualize and not self.vanilmode and self.it_in_epoch%10==0:
            utils.plotimage(xfk[0:1],dataX[0:1],'cifar10-arrow',"e"+str(self.curr_epoch)+"i"+str(self.it_in_epoch),self.run_name)


        return record


    def process_batch_test(self):

        record = {} #test statistics on current minibatch

        no_advers = self.vanilmode or (self.totit <= self.warmup_iters)

        # input data batch
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        bsize = dataX.shape[0]

        # Applying the random transform (multiplying input images by R)
        if not no_advers:
            Xp=to_gpu(torch.zeros_like(dataX))
            for k in range(3):
                cp=torch.matmul(dataX[:,k,:,:],self.randmat)
                Xp[:,k,:,:]=cp

        # Obtaining fake reconstrctuons and merging with real samples
        if not no_advers:
            xfk = self.lens(Xp)
            xj = torch.cat([dataX, xfk])
        else:
            xj = torch.cat([dataX])
        yfk = to_gpu(torch.ones_like(labels, dtype=torch.long))
        yfk[:] = self.nclass

        predj = self.classifier(xj)

        pred1 = predj[0:bsize, :] # classifier output on real samples

        l1 = ce_loss(pred1, labels) # classification loss

        if not no_advers:
            pred2 = predj[bsize:, :]
            l2 = ce_loss(pred2, yfk) # adverrsarial discrimination loss
        else:
            l2 = to_gpu(torch.zeros(1))


        loss_c = l1 + self.beta * l2 # total classification loss

        record['classifier_adv_loss'] = [l2]
        record['classifier_acc'] = [accuracy(pred1, labels, topk=(1,))[0]]
        record['classifier_acc_trun'] = [accuracy(pred1[:, 0:self.nclass], labels, topk=(1,))[0]] #accuracy without the last extra neuron
        record['classifier_ce_loss'] = [l1]
        
        #if not self.vanilmode:
        #    record['classifier_conf_ood']=[softmax(pred1)[:,self.nclass].mean()]
        #    record['confidence']=[torch.max(softmax(pred1[:,0:self.nclass]),dim=1)[0].mean()]
        #else:
        #    record['confidence']=[torch.max(softmax(pred1),dim=1)[0].mean()]
        if not no_advers:
            record['classifier_recal_fake'] = [accuracy(pred2, yfk, topk=(1,))[0]]
        #   record['confidence']=[torch.max(softmax(pred1),dim=1)[0].mean()]
        else:
            record['classifier_recal_fake'] = [torch.tensor(0)]
        record['classifier_loss'] = [loss_c]

        if no_advers:
            record['lens_loss'] = [torch.tensor(0)]
            return record

        # The lens loss based on either feature matching or cross-entropy

        if self.lensloss[0] == 'features':
            z = self.classifier(dataX, rep_block_index=self.lensloss[2])
            zf = self.classifier(xfk, rep_block_index=self.lensloss[2])

            if self.lensloss[1] == 'norm1':
                loss_g = torch.abs(z - zf).mean()
            if self.lensloss[1] == 'norm2':
                loss_g = rloss(z, zf)

        if self.lensloss[0] == 'ce':
            predf = self.classifier(xfk)
            loss_g = ce_loss(predf, labels)


        record['lens_loss'] = [loss_g]


        return record




softmax=Softmax(dim=1)
ce_loss= torch.nn.CrossEntropyLoss()

def rloss(r,x):

    return ((r-x)**2).mean()

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
    

    
 