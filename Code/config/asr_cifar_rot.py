# The Automatic Shortcut Removal method for Cifar10 (rotation prediction) experiments  

config = {}
config['algorithm']='ShortcutRemoval'

config['max_num_epochs'] = 30
config['batch_size'] = 128

config['is_downstream_task'] = False

#config['lambda']=640.0
config['lambda']=320.0
config['adv_loss_type']='least_likely'
config['n_class']=4

config_classify = {}
config_classify['n_outputs']=config['n_class']

config_lens = {}
config_lens['n_channels_in']=3

config['optim_params_classifier']={'optim_type': 'adam','lr':0.0001,'beta':(0.9, 0.999)}
config['optim_params_lens']={'optim_type': 'adam','lr':0.0001,'beta':(0.9, 0.999)}

networks = {}

networks['classifier'] = {'def_file': 'architectures.ResNet', 'pretrained': None, 'opt': config_classify}
networks['lens'] = {'def_file': 'architectures.Unet', 'pretrained': None, 'opt': config_lens}

config['networks'] = networks
config['trainables']= ['classifier','lens']
