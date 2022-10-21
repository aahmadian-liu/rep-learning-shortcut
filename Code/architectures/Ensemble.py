import torch
import torch.nn as nn
import torch.nn.functional as F
import architectures.ResNet as ResNet
import architectures.SmallMLP as SmallMLP

class Ensemble(nn.Module):

    def __init__(self, opt):

        super(Ensemble, self).__init__()

        opt['n_outputs']=opt['n_class']
        self.feature_ext=ResNet.create_model(opt)

        self.n_classifiers=opt['n_ensemble']
        self.n_class=opt['n_class']
        
        self.classifiers=nn.ModuleList()
        for i in range(self.n_classifiers):
            self.classifiers.append(SmallMLP.create_model(opt))

    def forward(self, x,return_h=False,rep_block_index=-1,flatten=False):

        if rep_block_index>-1:
            return self.feature_ext(x,rep_block_index=rep_block_index,flatten=flatten)

        y=torch.zeros(x.shape[0],self.n_class*self.n_classifiers)
        h=self.feature_ext(x,rep_block_index=5,flatten=True)

        for i in range(self.n_classifiers):
            y[:,i*self.n_class:(i+1)*self.n_class]=self.classifiers[i](h)

        if return_h:
            return y,h
        else:
            return y



def create_model(opt):
    return Ensemble(opt)