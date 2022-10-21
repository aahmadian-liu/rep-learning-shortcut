import matplotlib.pyplot as pyplot
import torchvision.transforms as transforms
#import biasdata
#Mask= biasdata.CreateGrMask()
from numpy.random import default_rng
rng = default_rng(0)
from dataloaders.cifar10_with_rotate import rotate_img
import numpy as np
import os
import globalconf


def plotimage(x_inp, x_recon,dataset, indx,title):

    x_re = x_recon.clone().squeeze()
    x_in = x_inp.clone().squeeze()

    if dataset=='cifar10-grad':
        mean_pix=[-1.11,-1.09,-0.93]
        std_pix=[0.88,0.90,0.86]

        x_re[0, :, :] = std_pix[0] * x_re[0, :, :] + mean_pix[0]
        x_re[1, :, :] = std_pix[1] * x_re[1, :, :] + mean_pix[1]
        x_re[2, :, :] = std_pix[2] * x_re[2, :, :] + mean_pix[2]

        x_in[0, :, :] = std_pix[0] * x_in[0, :, :] + mean_pix[0]
        x_in[1, :, :] = std_pix[1] * x_in[1, :, :] + mean_pix[1]
        x_in[2, :, :] = std_pix[2] * x_in[2, :, :] + mean_pix[2]

    if dataset.startswith('cifar10'):
        mean_pix = [r / 255.0 for r in [125.3, 123.0, 113.9]]
        std_pix = [r / 255.0 for r in [63.0, 62.1, 66.7]]

        x_re[0, :, :] = std_pix[0] * x_re[0, :, :] + mean_pix[0]
        x_re[1, :, :] = std_pix[1] * x_re[1, :, :] + mean_pix[1]
        x_re[2, :, :] = std_pix[2] * x_re[2, :, :] + mean_pix[2]

        x_in[0, :, :] = std_pix[0] * x_in[0, :, :] + mean_pix[0]
        x_in[1, :, :] = std_pix[1] * x_in[1, :, :] + mean_pix[1]
        x_in[2, :, :] = std_pix[2] * x_in[2, :, :] + mean_pix[2]


    inv_transform = transforms.Compose([lambda x: x.cpu().detach().numpy() * 255.0,
                                        lambda x: x.transpose(1, 2, 0).astype(np.uint8),
                                        ])

    x_in2 = inv_transform(x_in)
    x_re2 = inv_transform(x_re)

    # pyplot.figure()
    # pyplot.imshow(x_in2)

    # pyplot.figure()
    # pyplot.imshow(x_re2)
    # pyplot.show()
  
    pyplot.imsave(os.path.join(globalconf.work_dir,"Images_"+title,'x_recon_' + str(indx) + '.png'), x_in2)
    pyplot.imsave(os.path.join(globalconf.work_dir,"Images_"+title,'x_inp_' + str(indx) + '.png'), x_re2)
    print("image saved")



def puregrad(bsize):
    img1 = np.ones([32, 32, 3]) * Mask
    img2 = rotate_img(img1, 90)
    img3 = rotate_img(img1, 180)
    img4 = rotate_img(img1, 270)

    mean_pix = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std_pix = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_pix, std=std_pix)
    ])

    d = torch.stack([transform(img1), transform(img2), transform(img3), transform(img4)], dim=0)
    d = d.repeat(bsize // 4, 1, 1, 1)

    return d.float().cuda()


def purenoise(bsize):
    img1 = rng.random([32, 32, 3])
    img2 = rotate_img(img1, 90)
    img3 = rotate_img(img1, 180)
    img4 = rotate_img(img1, 270)

    mean_pix = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std_pix = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_pix, std=std_pix)
    ])

    d = torch.stack([transform(img1), transform(img2), transform(img3), transform(img4)], dim=0)
    d = d.repeat(bsize // 4, 1, 1, 1)

    return d.float().cuda()


def purecolor():
    digcolors = {0: (1, 1, 1), 1: (0.5, 0, 0), 2: (0, 0.5, 0), 3: (0, 0, 0.5), 4: (0.5, 0.5, 0), 5: (0, 0.5, 0.5),
                 6: (0.5, 0, 0.5), 7: (0.5, 0.5, 0.5), 8: (1, 0, 0.5), 9: (1, 0.5, 1)}

    imgs = []

    transform = transforms.Compose([transforms.ToTensor()])

    for i in range(0, 10):
        img = np.ones([28, 28, 3])
        for j in range(0, 3):
            img[:, :, j] = digcolors[i][j]
        imgs.append(transform(img))

    d = torch.stack(imgs, dim=0)
    return d.float().cuda()


def showgradnorm(params):
    gr = 0
    for p in params:
        gr += (p.grad ** 2).sum()
        gr = torch.sqrt(gr)

    print('gr', gr)


def weightsheatmap(classlayer, iternum):
    for p in classlayer.parameters():
        weights = p.cpu().detach().numpy()
        break

    weights = np.abs(weights)

    for i in range(0, weights.shape[0]):
        w = weights[i, :]
        wm = w.mean()
        wv = w.std()
        # u=np.abs(w-wm)/wv
        # w[u>1]=1
        # w[u<1]=0
        # weights[i,:]=(weights[i,:]-wm)/wv

    fig = pyplot.figure()
    # pyplot.imshow(weights[0:3,:],cmap='Blues')
    pyplot.hist(weights[2, :])

    fig.savefig("weightplots/iter" + str(iternum) + ".png")