"""Define a generic class for training and testing learning algorithms."""

import os
import os.path
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim

import utils
import datetime
import logging
import json
import globalconf

from pdb import set_trace as breakpoint


class Algorithm():
    def __init__(self, opt,run_name):

        self.opt = opt
        self.run_name = run_name

        self.set_log_file_handler()
        self.logger.info('Algorithm options %s' % opt)

        self.init_all_networks()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}
        
        if globalconf.gpu_mode:
            self.load_to_gpu()

        self.keep_best_model_metric_name = opt['best_metric'] if ('best_metric' in opt) else None

        self.logcurves_train=dict()
        self.logcurves_train_epoch = dict()
        self.logcurves_validation = dict()

        self.downstream=opt['is_downstream_task']
        self.best_acc=0
        self.best_acc_at=0
        self.bacn=0


    def update_log_curves(self,stats,curve,file_title):

        for key in stats.values.keys():
            if key not in curve.keys():
                curve[key] = []

            av=stats.average()[key]
            if len(av)==1:
                curve[key].append(str(av[0]))
            else:
                curve[key].append(str(av))
        
        if self.downstream:
            curve['best_acc']=str(self.best_acc)
            curve['best_acc_at']=str(self.best_acc_at)

        if not file_title=='':
            with open(os.path.join(globalconf.work_dir,self.run_name,file_title+'.json'), 'w') as f:
                json.dump(curve, f)
                print("loss curves saved")


    def save_models(self,epoch):

        for net in self.networks.keys():
            if net in self.opt['trainables']:
                torch.save(self.networks[net].state_dict(), os.path.join(globalconf.work_dir,self.run_name,'saved_' + net + '.pt'))
                if not self.downstream:
                    torch.save(self.optimizers[net].state_dict(), os.path.join(globalconf.work_dir,self.run_name,'saved_optim_' + net + '.pt'))
            
                if (epoch+1) in globalconf.save_on_epochs:
                    torch.save(self.networks[net].state_dict(),os.path.join(globalconf.work_dir, self.run_name, 'saved_' + net + "_epoch" + str(epoch+1) + '.pt'))

        print("network(s) saved to disk")


    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)

        self.vis_dir = os.path.join(directory_path,'visuals')
        if (not os.path.isdir(self.vis_dir)):
            os.makedirs(self.vis_dir)

        self.preds_dir = os.path.join(directory_path,'preds')
        if (not os.path.isdir(self.preds_dir)):
            os.makedirs(self.preds_dir)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(globalconf.work_dir, self.run_name)
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)

        now_str = datetime.datetime.now().__str__().replace(' ','_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

    def init_all_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            self.logger.info('Set network %s' % key)
            def_file = val['def_file']
            net_opt = val['opt']
            if key in self.opt['trainables']:
                self.optim_params[key] = self.opt['optim_params_'+key]
            pretrained_path = val['pretrained'] if ('pretrained' in val) else None
            self.networks[key] = self.init_network(def_file, net_opt, pretrained_path, key)

    def init_network(self, net_def_file, net_opt, pretrained_path, key):
        self.logger.info('==> Initiliaze network %s from %s with opts: %s' % (key, net_def_file, net_opt))

        network = importlib.import_module(net_def_file).create_model(net_opt)

        if pretrained_path != None:
            self.load_pretrained(network, pretrained_path)

        return network

    def load_pretrained(self, network, pretrained_path):
        self.logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))

        assert(os.path.isfile(pretrained_path))
        pretrained_model = torch.load(pretrained_path)
        if pretrained_model.keys() == network.state_dict().keys():
            network.load_state_dict(pretrained_model)
        else:
            self.logger.info('==> WARNING: network parameters in pre-trained file %s do not strictly match' % (pretrained_path))
            for pname, param in network.named_parameters():
                if pname in pretrained_model:
                    self.logger.info('==> Copying parameter %s from file %s' % (pname, pretrained_path))
                    param.data.copy_(pretrained_model['network'][pname])

    def init_all_optimizers(self):
        self.optimizers = {}

        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None:
                self.optimizers[key] = self.init_optimizer(self.networks[key], oparams, key)
            
            if self.opt['networks'][key]['pretrained'] !=None:
                self.optimizers[key].load_state_dict(torch.load(self.opt['networks'][key]['optim_pretrained']))

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for netwotk: %s'
            % (optim_type, optim_opts, key))
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                        betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=optim_opts['momentum'],
                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)

        return optimizer

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        if ctype=='BCEWithLogitsLoss':
            return getattr(nn, ctype)(pos_weight=torch.Tensor([copt]))
        else:
            return getattr(nn, ctype)(copt)

    def load_to_gpu(self):
        for key, net in self.networks.items():
            self.networks[key] = net.cuda()

        #for key, criterion in self.criterions.items():
        #    self.criterions[key] = criterion.cuda()

        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()

    def save_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue
            self.save_network(key, epoch, suffix=suffix)
            self.save_optimizer(key, epoch, suffix=suffix)

    def load_checkpoint(self, epoch, train=True, suffix=''):
        self.logger.info('Load checkpoint of epoch %d' % (epoch))

        for key, net in self.networks.items(): # Load networks
            if self.optim_params[key] == None: continue
            self.load_network(key, epoch,suffix)

        if train: # initialize and load optimizers
            self.init_all_optimizers()
            for key, net in self.networks.items():
                if self.optim_params[key] == None: continue
                self.load_optimizer(key, epoch,suffix)

        self.curr_epoch = epoch

    def delete_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue

            filename_net = self._get_net_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_net): os.remove(filename_net)

            filename_optim = self._get_optim_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_optim): os.remove(filename_optim)

    def save_network(self, net_key, epoch, suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'network': self.networks[net_key].state_dict()}
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'optimizer': self.optimizers[net_key].state_dict()}
        torch.save(state, filename)

    def load_network(self, net_key, epoch,suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.networks[net_key].load_state_dict(checkpoint['network'])

    def load_optimizer(self, net_key, epoch,suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])

    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_optim_epoch'+str(epoch))

    def solve(self, data_loader_train, data_loader_test,data_loader_train_2=None):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        eval_stats  = {}
        train_stats = {}
        self.init_record_of_best_model()

        #self.save_models(self.curr_epoch)
        #print("random model saved")
        #return

        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            print('Training epoch [%d / %d]' % (self.curr_epoch+1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            self.update_log_curves(train_stats,self.logcurves_train_epoch,'training_curves_epoch')

            self.logger.info('==> Training stats: %s' % (train_stats))
            #if start_epoch != self.curr_epoch: # delete the checkpoint of the previous epoch
            #    self.delete_checkpoint(self.curr_epoch)
            
            self.save_models(self.curr_epoch)

            if data_loader_test is not None:
                eval_stats = self.evaluate(data_loader_test)
                
                if self.downstream:
                    lastacc=eval_stats.average()['acc'][0]
                    if lastacc>self.best_acc:
                        self.best_acc=lastacc
                        self.best_acc_at=self.curr_epoch+1
                        torch.save(self.networks['ds_classifier'].state_dict(), os.path.join(globalconf.work_dir,self.run_name,'saved_ds_classifier_bestval.pt'))
                    print("Best acc:",self.best_acc, " (using rep block."+str(self.opt['representation_block'])+")")

                self.logger.info('==> Evaluation stats: %s' % (eval_stats))
                #self.keep_record_of_best_model(eval_stats, self.curr_epoch)
                self.update_log_curves(eval_stats,self.logcurves_validation,'validation_curves')

                #if self.downstream:
                #    acn=self.cum_acc/self.cum_acc_num
                #    print("test acc precise: ",acn)
                #    self.cum_acc=0
                #    self.cum_acc_num=0
                #    self.bacn=max(acn,self.bacn)
                #    print('best acc precise: ',self.bacn )


    def run_train_epoch(self, data_loader, epoch):
        self.logger.info('Training: %s' % self.run_name)

        for key, network in self.networks.items():
            if key in self.opt['trainables']: network.train()
            else: network.eval()

        #disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 50
        disp_step=1
        train_stats = utils.DAverageMeter()
        train_stats_b = utils.DAverageMeter()
        #self.bnumber = len(data_loader())
        
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx
            rec = self.train_step(batch)
            train_stats_b.update(rec)
            train_stats.update(rec)

            if (idx+1) % globalconf.display_every_iters == 0:
                print('stats on batch:', train_stats_b)
                self.update_log_curves(train_stats_b,self.logcurves_train,'training_curves')
            else:
                self.update_log_curves(train_stats_b, self.logcurves_train, '')

            train_stats_b.reset()

            if globalconf.save_every_iters>0 and (idx+1) % globalconf.save_every_iters == 0:
                self.save_models(epoch)


        return train_stats



    def evaluate(self, dloader):
        self.logger.info('Evaluating: %s' % self.run_name)

        self.dloader = dloader
        self.dataset_eval = dloader.dataset
        #self.logger.info('==> Dataset: %s [%d images]' % (dloader.dataset.name, len(dloader)))
        for key, network in self.networks.items():
            network.eval()

        eval_stats = utils.DAverageMeter()
        #self.bnumber = len(dloader())
        
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        #self.logger.info('==> Results: %s' % eval_stats.average())
        print("Test stats:",eval_stats.average())

        return eval_stats

    def adjust_learning_rates(self, epoch):
        optim_params_filtered = {k:v for k,v in self.optim_params.items()
            if (v != None and ('LUT_lr' in v))}

        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            print('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if (metric_name not in eval_stats):
                raise ValueError('The provided metric {0} for keeping the best model is not computed by the evaluation routine.'.format(metric_name))
            metric_val = eval_stats[metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(self.curr_epoch+1, suffix='.best')
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch+1, suffix='.best')
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s' % (metric_name, self.best_epoch+1, self.best_stats))


    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}
