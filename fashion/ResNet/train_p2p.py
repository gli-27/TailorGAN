#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:15:51 2019

@author: justintian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:33:26 2019

@author: justintian
"""

import os
import sys
import cycleNet
from p2pDataGen import p2pFashionDataset
import p2pNetworks
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import itertools
import random
from param import get_params
os.environ['CUDA_VISIBLE_DEVICES']='0'

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


G= p2pNetworks.generator(64)
D= p2pNetworks.discriminator(64)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G.train()
D.train()

BCE_loss=nn.BCELoss().cuda()
MSE_loss=nn.MSELoss().cuda()
L1_loss=nn.L1Loss().cuda()

# loss
BCE_loss = nn.BCELoss().cuda()
L1_loss = nn.L1Loss().cuda()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=3e-4, betas=(0.5, 0.999))


params=get_params()
num_epochs=60
batch_size=params['batch_size']

dataset=p2pFashionDataset(params)
loader=DataLoader(dataset, batch_size=batch_size)
    
fakeA_store = ImagePool(50)
fakeB_store = ImagePool(50)




for epoch in range(num_epochs):
    for i, data in enumerate(loader):
        D.zero_grad()
        x,y=data
        x, y = Variable(x.cuda()), Variable(y.cuda())
        
        D_result = D(y).squeeze()
        D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))

        G_result = G(x)
        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))
        
        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()
        
        G.zero_grad()

        G_result = G(x)
        D_result = D(G_result).squeeze()

        G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + 100.0* L1_loss(G_result, y)
        G_train_loss.backward()
        G_optimizer.step()

        # print results
        print('At epoch {}, step {}, loss_D:{:.4f}, loss_G:{:.4f}\n'.format(epoch, i, D_train_loss, G_train_loss))
        if i>0 and (i+1) % 2000 == 0:
                torch.save(G.state_dict(), './model/p2pG_'+str(epoch)+'.ckpt')
                #torch.save(G_B.state_dict(), './model/cycleGB12_'+str(epoch)+'.ckpt')
        
