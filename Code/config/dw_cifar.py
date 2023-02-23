# Downstream classification for Cifar10 experiments, used after the ARE, SD and DE methods
# To use with DE, change the feature extractor def_file to 'architectures.Ensemble'


config = {}
config['algorithm']='DownstreamClassification'

config['max_num_epochs'] = 30
config['batch_size'] = 64

config['is_downstream_task'] = True
config['has_feature_extractor'] = True
config['has_lens'] = False

networks = {}

net_opt_c={}
net_opt_c['n_hidden']=100
net_opt_c['batchnorm']=True
net_opt_c['n_class']=10

networks['feature_extractor'] = {'def_file': 'architectures.ResNet', 'pretrained': None, 'opt': None}
#for DE:
#networks['feature_extractor'] = {'def_file': 'architectures.Ensemble', 'pretrained': None, 'opt': None}
config['representation_block']= 2
config['flat_features']=True

#number of dimensions of the representation vector
inputdims=[4096,-1,8192,-1,-1,2048] # for rep blocks 0,2,5
net_opt_c['n_input_dims']=inputdims[config['representation_block']]


networks['ds_classifier'] = {'def_file': 'architectures.LinearMLP', 'pretrained': None, 'opt': net_opt_c}
config['optim_params_ds_classifier']={'optim_type': 'adam','lr':0.001,'beta':(0.9, 0.999)}

config['networks'] = networks
config['trainables']= ['ds_classifier']

