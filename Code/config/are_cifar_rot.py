# The proposed Adversarial Representation Enhancement (ARE) method for CIFAR10 (rotation prediction) experiments  

config = {}
config['algorithm']='AdvRepEnhance'
config['is_downstream_task'] = False

config['max_num_epochs'] = 30
config['batch_size'] = 128

config['vanilla_classifier_mode']=False

config['n_class']=4
config['image_size']=[32,32]

config['beta']=0.5
config['warmup_iterations']=2
config['steps_classifier']=1
config['steps_lens']=1
config['lens_loss_config']=('features','norm2',2)
config['grad_clip_max']=150

config_classify = {}
if not config['vanilla_classifier_mode']:
    config_classify['n_outputs']=config['n_class']+1
else:
    config_classify['n_outputs']=config['n_class']

config_lens = {}
config_lens['n_channels_in']=3

networks = {}

networks['classifier'] = {'def_file': 'architectures.ResNet', 'pretrained': None, 'opt': config_classify}

networks['lens'] = {'def_file': 'architectures.Unet', 'pretrained': None, 'opt': config_lens}

config['optim_params_classifier']={'optim_type': 'adam','lr':0.0002,'beta':(0.9, 0.999)}
config['optim_params_lens']={'optim_type': 'adam','lr':0.002,'beta':(0.9, 0.999)}

config['networks'] = networks
if not config['vanilla_classifier_mode']:
    config['trainables']= ['classifier','lens']
else:
    config['trainables']= ['classifier']
