import torch
import torch.nn as nn


class LinearMLP(torch.nn.Module):
    def __init__(self,opt):
        super(LinearMLP, self).__init__()
        self.nhidden=opt['n_hidden']
        self.bn=opt['batchnorm']

        outdim=opt['n_class']
        indim=opt['n_input_dims']

        if self.bn and self.nhidden>0:
            self.layers= nn.Sequential(nn.Linear(indim,self.nhidden),nn.BatchNorm1d(self.nhidden),nn.Linear(self.nhidden,outdim))

        if not self.bn and self.nhidden>0:
            self.layers= nn.Sequential(nn.Linear(indim,self.nhidden),nn.Linear(self.nhidden,outdim))

        if self.nhidden==0:
            self.layers= nn.Sequential(nn.Linear(indim,outdim))


    def forward(self, x):
        outputs = self.layers(x)

        outputs=torch.squeeze(outputs)
        return outputs


def create_model(opt):
    return LinearMLP(opt)