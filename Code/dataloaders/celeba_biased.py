import globalconf
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch



input_att1_index=9  #blond hair
input_att2_index=8 #black hair
target1_att_index=20 #male
target2_att_index=31 #smiling

dir=globalconf.data_dir+'/celeb/'



class CelebA_Loader(object):

    def __init__(self,partition,mode,batch_size,num_workers=1,num_imgs_per_cat=None,imgs_cat_part=None):

        assert(mode in {'biased_gender','biased_color','balanced_smile','balanced_cg','original_gender'})
        self.mode=mode

        self.dataset = CelebA(dir,partition,download=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize([64,64])]))

        self.len_data=len(self.dataset)

        self.batch_size=batch_size
        self.num_workers = num_workers

        self.minor_x=torch.zeros(180,3,64,64)
        self.minor_y=torch.zeros(180)

        if num_imgs_per_cat is not None:
            self.keep_samples=num_imgs_per_cat
            self.keep_part=imgs_cat_part
        else:
            self.keep_samples=-1


    def build_minor_set(self):
        loader=DataLoader(self.dataset,batch_size=1000,num_workers=self.num_workers)
        ind=0
        print('building the minority set...')
        for x,y in loader:
            s= (y[:,target1_att_index]==1) & (y[:,input_att1_index]==1)
            x_s=x[s,:]
            y_s=y[s,target1_att_index]

            bsize=y_s.shape[0]
            self.minor_x[ind:(ind+bsize),:]=x_s
            self.minor_y[ind:(ind+bsize)]=y_s
            ind+=bsize         


    def get_generator_bias(self):

        loader=iter(DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False))

        while True:
            d=next(loader,None)

            if d is not None:
                x,y=d[0],d[1]

                s= ( (y[:,target1_att_index]==0) & (y[:,input_att1_index]==1) ) | ( (y[:,target1_att_index]==1) & (y[:,input_att2_index]==1) )
                x_s=x[s,:]
                y_s=y[s,target1_att_index]

                yield (x_s,y_s)
            else:
                break
    
    def get_generator_bias_inv(self):

        loader=iter(DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False))

        while True:
            d=next(loader,None)

            if d is not None:
                x,y=d[0],d[1]
            
                s= ( (y[:,target1_att_index]==0) & (y[:,input_att1_index]==1) ) | ( (y[:,target1_att_index]==1) & (y[:,input_att2_index]==1) )
                x_s=x[s,:]
                y_s=y[s,input_att1_index]

                yield (x_s,y_s)
            else:
                break

    def get_generator_minor(self):

        while True:

            yield (self.minor_x,self.minor_y)
            break

    def get_generator_original(self):

        loader=iter(DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False))

        while True:
            d=next(loader,None)

            if d is not None:
                x,y=d[0],d[1]

                yield (x,y[:,target1_att_index])
            else:
                break


    def get_generator_balance_cg(self):

        loader=DataLoader(self.dataset,batch_size=2000,num_workers=self.num_workers,shuffle=False)

        for x,y in loader:
        
            s1= (y[:,target1_att_index]==1) & (y[:,input_att1_index]==1) 
            x_n_1=x[s1,:]
            y_n_1=y[s1,target1_att_index]

            s2 = (y[:,target1_att_index]==0) & (y[:,input_att2_index]==1)
            x_n_2=x[s2,:]
            y_n_2=y[s2,target1_att_index]

            s3= (y[:,target1_att_index]==0) & (y[:,input_att1_index]==1) 
            x_b_1=x[s3,:]
            y_b_1=y[s3,target1_att_index]

            s4 = (y[:,target1_att_index]==1) & (y[:,input_att2_index]==1)
            x_b_2=x[s4,:]
            y_b_2=y[s4,target1_att_index]

            if (y_n_1.shape[0]==0 or y_n_2.shape[0]==0):
                break
            
            
            m=min(y_n_1.shape[0],y_n_2.shape[0],y_b_1.shape[0],y_b_2.shape[0])

            if self.keep_samples>0:
                m=self.keep_samples

            x_n_1=x_n_1[0:m,:]
            y_n_1=y_n_1[0:m]
            x_n_2=x_n_2[0:m,:]
            y_n_2=y_n_2[0:m]
            x_b_1=x_b_1[0:m,:]
            y_b_1=y_b_1[0:m]
            x_b_2=x_b_2[0:m,:]
            y_b_2=y_b_2[0:m]
            
            x_s=torch.cat([x_n_1,x_n_2,x_b_1,x_b_2])
            y_s=torch.cat([y_n_1,y_n_2,y_b_1,y_b_2])

            print('male',y_s[y_s==1].shape[0])
            print('female',y_s[y_s==0].shape[0])

            yield (x_s,y_s)

            if self.keep_samples>0:
                if x_s.shape[0]<4*m:
                    print("num_examples_per_class not attained")
                break


    def get_generator_balance_smile(self):

        loader=DataLoader(self.dataset,batch_size=500,num_workers=self.num_workers,shuffle=False)

        kp=0

        for x,y in loader:

            if self.keep_samples>0:
                if kp<self.keep_part:
                    kp+=1
                    continue

            s1= (y[:,target2_att_index]==1)
            x_n_1=x[s1,:]
            y_n_1=y[s1,target2_att_index]

            s2 = (y[:,target2_att_index]==0)
            x_n_2=x[s2,:]
            y_n_2=y[s2,target2_att_index]
            
            #print("n1",y_n_1.shape[0])
            #print("n2",y_n_2.shape[0])

            m=min(y_n_1.shape[0],y_n_2.shape[0])

            if self.keep_samples>0:
                m=self.keep_samples
                
            x_n_1=x_n_1[0:m,:]
            y_n_1=y_n_1[0:m]
            x_n_2=x_n_2[0:m,:]
            y_n_2=y_n_2[0:m]
            
            x_s=torch.cat([x_n_1,x_n_2])
            y_s=torch.cat([y_n_1,y_n_2])

            print('bsize',x_s.shape[0])

            yield (x_s,y_s)

            if self.keep_samples>0:
                if x_s.shape[0]<2*m:
                    print("num_examples_per_class not attained")
                break



    def __call__(self,epoch=0):
        if self.mode=='biased_gender':
            return self.get_generator_bias()
        elif self.mode=='biased_color':
            return self.get_generator_bias_inv() 
        elif self.mode=='balanced_cg':
            return self.get_generator_balance_cg()  
        elif self.mode=='balanced_smile':
            return self.get_generator_balance_smile()    
        elif self.mode=='original_gender':
            return self.get_generator_original() 

    def __len__(self):
        return self.len_data / self.batch_size




def Celeba_Original_Loader(partition,batch_size):

    celebdata= CelebA(dir,partition,download=True,transform=transforms.ToTensor())
    loader=DataLoader(celebdata,batch_size=batch_size)
    return loader

