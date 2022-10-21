import torch
import torch.nn as nn


class SmallMLP(torch.nn.Module):
    def __init__(self,opt):
        super(SmallMLP, self).__init__()
        self.nhidden=opt['n_hidden']

        outdim=opt['n_class']
        indim=opt['n_input_dims']

        self.layers= nn.Sequential(nn.Linear(indim,self.nhidden[0]),nn.ReLU(),nn.Linear(self.nhidden[0],self.nhidden[1]),nn.ReLU(),nn.Linear(self.nhidden[1],outdim))


    def forward(self, x):
        outputs = self.layers(x)

        #outputs=torch.squeeze(outputs)
        return outputs


def create_model(opt):
    return SmallMLP(opt)