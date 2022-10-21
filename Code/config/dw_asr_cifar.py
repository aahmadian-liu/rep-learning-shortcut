# Downstream classification after Automatic Shortcut Removal method for cifar10 experiments

config = {}
config['algorithm']='DownstreamClassification'

config['max_num_epochs'] = 30
config['batch_size'] = 64

config['is_downstream_task'] = True
config['has_feature_extractor'] = True
config['has_lens'] = True
config['concat_mode']=True

networks = {}

net_opt_c={}
net_opt_c['n_hidden']=100
net_opt_c['batchnorm']=True
net_opt_c['n_class']=10

networks['feature_extractor'] = {'def_file': 'architectures.ResNet', 'pretrained': None, 'opt': None}
config['representation_block']= 2
config['flat_features']=True
#net_opt_c['n_input_dims']=4096*2
#net_opt_c['n_input_dims']=2048*2
net_opt_c['n_input_dims']=8192*2


networks['ds_classifier'] = {'def_file': 'architectures.LinearMLP', 'pretrained': None, 'opt': net_opt_c}

config['optim_params_ds_classifier']={'optim_type': 'adam','lr':0.001,'beta':(0.9, 0.999)}

networks['lens'] = {'def_file': 'architectures.Unet', 'pretrained': None, 'opt':{'n_channels_in':3}}

config['networks'] = networks
config['trainables']= ['ds_classifier']
