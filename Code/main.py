# Implementation of the paper Enhancing Representation Learning with Deep Classifiers in Presence of Shortcut (A.Ahmadian and F.Lindsten)
# amirhossein.ahmadian@liu.se
# Oct 2022
# Part of the code is based on https://github.com/gidariss/FeatureLearningRotNet


import globalconf
import argparse
import os
import torch
import random
import numpy as np
import datetime
import json

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

import importlib
import datasets


parser = argparse.ArgumentParser()
parser.add_argument('config_algorithm', type=str, help='the configuration script of the algorithm that will be run (title of a script in config directory)')
parser.add_argument('dataset',type=str, help='dataset name for training and/or test (as defined in the the datasets.py)')
parser.add_argument('data_mode_train',type=str, help='the mode (i.e., shortcut bias) of the training data')
parser.add_argument('data_mode_test',type=str,help='the mode (i.e., shortcut bias) of the test data')
parser.add_argument('--feature_extractor_dir',type=str, required=False,default='',help='the path to the directory of a pre-trained model (feature extractor networks) used for obtaining representations; applies to downstream training')
parser.add_argument('--feature_extractor_config',type=str, required=False,default='',help='the configuration script of the upstream training algorithm; to be used with feature_extractor_dir')
#parser.add_argument('--extractor_vanilla',default=False, action='store_true',help='whether the feature extractor model has a vanilla structure; can be used with feature_extractor_dir')
parser.add_argument('--title',type=str, required=False,default='',help='title of this experiment, used for the results directory (the default title is based on algorithm and dataset)')
parser.add_argument('--evaluate',    default=False, action='store_true',help='only evaluation on the test data (no training)')
parser.add_argument('--load_model_path', type=str, required=False,default='',help='the directory from which pre-trained networks are loaded; used if the main model should be intialized from file')
parser.add_argument('--working_dir',  type=str, required=False, default='',help='parent directory for saving results (overrides the global config)')
parser.add_argument('--cuda'  ,      default=False,action='store_true', help='enables using cuda GPU (overrides the global config)')
parser.add_argument('--override_config', type=str, required=False,default='', help='can be used to override the current configurations in config_algorithm; should be in the format c1=x,c2=y')
args = parser.parse_args()


# Setting the default configs
if not args.working_dir=='':
    globalconf.work_dir=args.working_dir
if args.title=='':
    if args.data_mode_train==args.data_mode_test:
        args.title=args.config_algorithm + "_" + args.data_mode_train
    else:
        args.title=args.config_algorithm + "_" + args.data_mode_train + "_" + args.data_mode_test
if args.cuda==True:
    globalconf.gpu_mode=True

# Loading the main configuration script
config=importlib.import_module("config."+args.config_algorithm).config

if not args.override_config=='':
    for s in args.override_config.split(','):
        sl=s.split('=')[0]
        sr=s.split('=')[1]
        config[sl]=eval(sr)

# Loading the feature extractor model and its configuration, if required 
if not args.feature_extractor_config=='':
    config_extractor=importlib.import_module("config."+args.feature_extractor_config)

# Creating the directories for results and figures
if not os.path.isdir(os.path.join(globalconf.work_dir,args.title)):
    os.makedirs(os.path.join(globalconf.work_dir,args.title))

if config['algorithm'].startswith('DownstreamClassification') or config['algorithm'].startswith('DiverseEnsemble'):
    globalconf.visualize=False
if globalconf.visualize:
    if not os.path.isdir(os.path.join(globalconf.work_dir,"Images_"+args.title)):
        os.makedirs(os.path.join(globalconf.work_dir,"Images_"+args.title))
    else:
        input("images directory not empty..")

for con in config['networks'].values():
    if not con['pretrained'] is None:
        print("using weights loaded from ",con['pretrained'])

config['start_time'] = str(datetime.datetime.now())
config['command']=str(args)

# Setting train and test datasets and the corresponding data loaders
data_loader_train,data_loader_test=datasets.get_data_train_test(args.dataset,args.data_mode_train,args.data_mode_test,config)

# Setting the pretrained networks paths if the model should be loaded
if not args.load_model_path=='':
    for net in config['networks'].keys():
        config['networks'][net]['pretrained']=os.path.join(args.load_model_path, "saved_"+net+".pt")
        config['networks'][net]['optim_pretrained']=os.path.join(args.load_model_path, "saved_optim_"+net+".pt")


# Writing all the used configs to a json file
with open(os.path.join(globalconf.work_dir, args.title, 'configs.json'), 'w') as f:
    config['save_on_epochs']=globalconf.save_on_epochs
    json.dump(config, f)


# Loading the required modules and networks for the specified algorithm

if config['algorithm'].startswith('AdvLensRan'):

    algorithm= importlib.import_module("algorithms."+config['algorithm']).AdvLensRan(config,args.title)

elif config['algorithm'].startswith('DiverseEnsemble'):

    algorithm= importlib.import_module("algorithms."+config['algorithm']).DiverseEnsemble(config,args.title)

elif config['algorithm'].startswith('BasicClassification'):

    algorithm= importlib.import_module("algorithms."+config['algorithm']).BasicClassification(config,args.title)

elif config['algorithm'].startswith('DownstreamClassification'):

    config['networks']['feature_extractor']['opt']=config_extractor.config_classify
    config['networks']['feature_extractor']['pretrained']= os.path.join(args.feature_extractor_dir,"saved_classifier.pt")
    if config_extractor.config['algorithm']=='DiverseEnsemble':
        config['networks']['feature_extractor']['pretrained']= os.path.join(args.feature_extractor_dir,"saved_ensemble.pt")
    if 'lens' in config['networks']:
        config['networks']['lens']['pretrained']= os.path.join(args.feature_extractor_dir,"saved_lens.pt")

    algorithm= importlib.import_module("algorithms."+config['algorithm']).DownstreamClassifier(config,args.title)

elif config['algorithm'].startswith('ShortcutRemoval'):

    algorithm= importlib.import_module("algorithms."+config['algorithm']).ShortcutRemoval(config,args.title)
else:
    raise(ValueError("Undefined algorithm"))


# Running the training/evaluation algorithm 
if not args.evaluate:
    algorithm.solve(data_loader_train, data_loader_test)
else:
    algorithm.evaluate(data_loader_test)
