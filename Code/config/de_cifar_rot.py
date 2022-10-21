# The Diverse Ensemble method for Cifar10 (rotation prediction) experiments  

config = {}
config['algorithm']='DiverseEnsemble'

config['max_num_epochs'] = 30
config['batch_size'] = 128
config['n_class']=4

config['is_downstream_task'] = False

config['lambda_div']=1000

config_classify = {}
config_classify['n_class']=config['n_class']
config_classify['n_ensemble']=32

config_classify['n_hidden']=[512,512]
config_classify['n_input_dims']=2048

networks = {}

networks['ensemble'] = {'def_file': 'architectures.Ensemble', 'pretrained': None, 'opt': config_classify}
config['optim_params_ensemble']={'optim_type': 'adam','lr':0.0002,'beta':(0.9, 0.999)}

config['networks'] = networks
config['trainables']= ['ensemble']