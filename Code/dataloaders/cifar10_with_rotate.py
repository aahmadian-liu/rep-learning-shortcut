
import torch
import torch.utils.data as data
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.dataloader import default_collate
import numpy as np
from os.path import join as joinpath
import globalconf

from pdb import set_trace as breakpoint



dataset_dir = joinpath(globalconf.data_dir,'cifar10')




class Dataset(data.Dataset):
    def __init__(self, split,dataset_name="cifar10",num_imgs_per_cat=None):

        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split

        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name=='cifar10':

            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]

            transform = []
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](dataset_dir, train=self.split=='train',download=True, transform=self.transform)

        else:
            raise ValueError('Not recognized dataset')
        
        if num_imgs_per_cat is not None:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)
        
    
    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))
   
        if self.dataset_name=='cifar10':
            #labels = self.data.test_labels if (self.split=='test') else self.data.train_labels
            labels=self.data.targets
            #data = self.data.test_data if (self.split=='test') else self.data.train_data
            data=self.data.data
            label2ind = buildLabelIndex(labels)
            all_indices = []
            for cat in label2ind.keys():
                label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
                all_indices += label2ind[cat]
            all_indices = sorted(all_indices)
            data = data[all_indices]
            labels = [labels[idx] for idx in all_indices]
            #if self.split=='test':
            #    self.data.test_labels = labels
            #    self.data.test_data = data
            #else:
            #    self.data.train_labels = labels
            #    self.data.train_data = data

            self.data.data=data
            self.data.targets=labels

            label2ind = buildLabelIndex(labels)
            for k, v in label2ind.items(): 
                assert(len(v)==num_imgs_per_cat)
        else:
            raise ValueError('Not recognized dataset {0}')


    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2))).copy()
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DataLoader(object):
    def __init__(self,dataset,batch_size,rotation_mode,epoch_size=None,num_workers=1,shuffle=True,synthetic_bias=None,oodmode=False,normalize_standard=True):

        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = rotation_mode
        self.num_workers = num_workers

        self.synbias=synthetic_bias
        self.oodmode=oodmode

        self.normalize_standard=normalize_standard

        if not (self.synbias is None):
            assert(self.synbias in {'grad','marker','arrow'})

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix

        if normalize_standard and not self.synbias=='grad':
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean_pix, std=std_pix)])
            self.inv_transform = transforms.Compose([Denormalize(mean_pix, std_pix),lambda x: x.numpy() * 255.0,lambda x: x.transpose(1,2,0).astype(np.uint8)])

        if normalize_standard and self.synbias=='grad':
            mean_pix_b=[-1.11,-1.09,-0.93]
            std_pix_b=[0.88,0.90,0.86]

            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean_pix, std=std_pix),transforms.Normalize(mean=mean_pix_b, std=std_pix_b)])
        
        if not normalize_standard:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x*2-1)])


    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]

                if self.synbias=='marker':
                    img0=Drawmarker(img0)
                if self.synbias=='arrow':
                    img0=Drawarrow(img0)
                if self.synbias=='grad':
                    img0=np.array(Mask*img0,np.uint8)
                   
                if self.oodmode:
                   img0=np.array(Mask*np.ones_like(img0)*255,np.uint8)

                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]

                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]

                if self.synbias=='marker':
                    img=Drawmarker(img)
                if self.synbias=='arrow':
                    img=Drawarrow(img)
                if self.synbias=='grad':
                    img=np.array(Mask*img,np.uint8)

                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)

        return iter(data_loader)


    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size


def GradMask():
    imsize = 32
    mask = np.zeros([imsize, imsize])
    for i in range(imsize):
        mask[i, :] = max(0, 1 - float(i) * 1.2 / imsize)
    mask = mask[:, :, np.newaxis]
    return mask

def Drawmarker(img):
    img2 = img.copy()
    img2[3:9, 3:6, :] = 255
    return img2

def Drawarrow(img):
    img2 = img.copy()
    pad=0

    img2[(5+pad):(12+pad), 5+pad, :] = 127
    img2[(6+pad), (4+pad):(7+pad), :] = 127
    img2[7+pad, (3+pad):(8+pad), :] = 127

    return img2

Mask= GradMask()


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds