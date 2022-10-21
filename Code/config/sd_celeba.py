# The Spectral Decoupling method / vanilla baseline for CelebA experiments  (use logits_penalty=0 to have vanilla method)

config = {}
config['algorithm']='BasicClassification'
config['is_downstream_task'] = False

config['max_num_epochs'] = 10
config['batch_size'] = 128

config['n_class']=2
config['image_size']=[64,64]

config['logits_penalty']=(0.04/2.0,'mean')
#config['logits_penalty']=(0,None)

config_classify = {}
config_classify['n_outputs']=config['n_class']


networks = {}

networks['classifier'] = {'def_file': 'architectures.ResNet', 'pretrained': None, 'opt': config_classify}

config['optim_params_classifier']={'optim_type': 'adam','lr':0.0001,'beta':(0.9, 0.999)}

config['networks'] = networks
config['trainables']= ['classifier']
