""" The interface to datasets """

from dataloaders import cifar10_with_rotate,celeba_biased
import globalconf

#Implemented datasets:
#
# 'cifar10' : cifar10 images either in self-supervised rotation prediction mode (4 classes) or downstream supervised mode (10 classes)
#    modes: 'clean' (original images), 'arrow' (an arrow added at the up-left corner), 'grad' (vertical gradient of brightness by multiplying a mask) 
# 'celeba' : CelebA face images, labeled with either gender or smile binary classes (see below)
#    modes: 'original_gender' (original dataset; gender attribute is the target), 'biased_gender' (only blond female and black-haired male; gender attribute is the target), 
#    'balanced_smile' (smiling attribute is the target; same number of data in the two classes)
#

def get_data_train_test(dataset,data_mode_train,data_mode_test,config_algorithm):

        if 'max_examples_per_class' in config_algorithm.keys() and config_algorithm['max_examples_per_class']>0:
                numpercat=config_algorithm['max_examples_per_class']
                catpart=config_algorithm['examples_per_class_part']
                print("Training examples per class ", numpercat)
        else:
                numpercat=None
                catpart=None

        if dataset=='cifar10':

                dataset_train = cifar10_with_rotate.Dataset(split='train')
                dataset_test = cifar10_with_rotate.Dataset(split='test')

                if data_mode_train=='clean':
                        data_mode_train=None
                if data_mode_test=='clean':
                        data_mode_test=None
                oodmode=(data_mode_test=='ood')

                normalize_meanvar= not (config_algorithm['algorithm']=='ShortcutRemoval')

                dloader_train = cifar10_with_rotate.DataLoader(
                        dataset=dataset_train,
                        batch_size=config_algorithm['batch_size'],
                        rotation_mode=not config_algorithm['is_downstream_task'],
                        num_workers=globalconf.num_workers,
                        shuffle=False,synthetic_bias=data_mode_train,normalize_standard=normalize_meanvar)

                dloader_test= cifar10_with_rotate.DataLoader(
                        dataset=dataset_test,
                        batch_size=config_algorithm['batch_size'],
                        rotation_mode=not config_algorithm['is_downstream_task'],
                        num_workers=globalconf.num_workers,
                        shuffle=False,synthetic_bias=data_mode_test,oodmode=oodmode,normalize_standard=normalize_meanvar)

        elif dataset == 'cmnist':

                dataset_train = colored_mnist.Dataset(split='train',num_imgs_per_cat=numpercat,imgs_cat_part=catpart)
                dataset_test = colored_mnist.Dataset(split='test')

                nobiasratio=config_algorithm['ratio_nonbias'] if 'ratio_nonbias' in config_algorithm else 0

                dloader_train = colored_mnist.DataLoader(
                        dataset=dataset_train,color_mode=data_mode_train,
                        batch_size=config_algorithm['batch_size'],
                        num_workers=globalconf.num_workers,
                        shuffle=False,ratio_nonbias=nobiasratio)

                dloader_test = colored_mnist.DataLoader(
                        dataset=dataset_test, color_mode=data_mode_test,
                        batch_size=config_algorithm['batch_size'],
                        num_workers=globalconf.num_workers,
                        shuffle=False)
                        
        elif dataset == 'celeba':
                dloader_train = celeba_biased.CelebA_Loader('train',data_mode_train,config_algorithm['batch_size'],globalconf.num_workers,num_imgs_per_cat=numpercat,imgs_cat_part=catpart)
                
                dloader_test = celeba_biased.CelebA_Loader('test',data_mode_test,config_algorithm['batch_size'],globalconf.num_workers)
                if data_mode_test=='minority':
                        dloader_test.build_minor_set()

        else:
                raise(Exception("Invalid dataset name"))


        return dloader_train,dloader_test
