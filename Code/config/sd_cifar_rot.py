# The Spectral Decoupling method / vanilla baseline for Cifar10 experiments  (use logits_penalty=0 to have vanilla method)

config = {}
config['algorithm']='BasicClassification'
config['is_downstream_task'] = False

config['max_num_epochs'] = 30
config['batch_size'] = 128

config['n_class']=4
config['image_size']=[32,32]

config['logits_penalty']=(0.01/2.0,'sum')
#config['logits_penalty']=(0,None)

config_classify = {}
config_classify['n_outputs']=config['n_class']


networks = {}

networks['classifier'] = {'def_file': 'architectures.ResNet', 'pretrained': None, 'opt': config_classify}

config['optim_params_classifier']={'optim_type': 'adam','lr':0.001,'beta':(0.9, 0.999)}

config['networks'] = networks
config['trainables']= ['classifier']