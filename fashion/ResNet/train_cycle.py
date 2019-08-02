#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:33:26 2019

@author: justintian
"""

import os
import sys
import cycleNet
from cycleDataGen import CycleFashionDataset
import cycleNet
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import itertools
import random
from param import get_params
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_id=2
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


G_A=cycleNet.generator(3,3,32,9)
G_B=cycleNet.generator(3,3,32,9)
D_A=cycleNet.generator(3,1,64)
D_B=cycleNet.generator(3,1,64)
G_A.weight_init(mean=0.0, std=0.02)
G_B.weight_init(mean=0.0, std=0.02)
D_A.weight_init(mean=0.0, std=0.02)
D_B.weight_init(mean=0.0, std=0.02)
G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()
G_A.train()
G_B.train()
D_A.train()
D_B.train()

BCE_loss=nn.BCELoss().cuda()
MSE_loss=nn.MSELoss().cuda()
L1_loss=nn.L1Loss().cuda()

G_optimizer=optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=1e-4, betas=(0.5,0.999))
D_A_optimizer = optim.Adam(D_A.parameters(), lr=5e-4, betas=(0.5,0.999))
D_B_optimizer = optim.Adam(D_B.parameters(), lr=5e-4, betas=(0.5,0.999))

params=get_params()
num_epochs=40
batch_size=params['batch_size']

dataset=CycleFashionDataset(params,0,1)
loader=DataLoader(dataset, batch_size=batch_size)
    
fakeA_store = ImagePool(50)
fakeB_store = ImagePool(50)




for epoch in range(num_epochs):
    for i, data in enumerate(loader):
        realA,realB=data
        realA,realB=Variable(realA.cuda()), Variable(realB.cuda())
        #train generator G
        G_optimizer.zero_grad()
        # generate real A to fake B
        fakeB = G_A(realA)
        D_A_result = D_A(fakeB)
        G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).cuda()))
        # reconstruct fake B to rec A; G_B(G_A(A))
        recA = G_B(fakeB)
        A_cycle_loss = L1_loss(recA, realA) * 10.0

        # generate real B to fake A; D_A(G_B(B))
        fakeA = G_B(realB)
        D_B_result = D_B(fakeA)
        G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).cuda()))
        # reconstruct fake A to rec B G_A(G_B(B))
        recB = G_A(fakeA)
        B_cycle_loss = L1_loss(recB, realB) * 10.0

        G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss
        G_loss.backward()
        G_optimizer.step()
        
        
        
        # train discriminator D_A
        D_A_optimizer.zero_grad()

        D_A_real = D_A(realB)
        D_A_real_loss = MSE_loss(D_A_real, Variable(torch.ones(D_A_real.size()).cuda()))

        # fakeB = fakeB_store.query(fakeB.data)
        fakeB = fakeB_store.query(fakeB)
        D_A_fake = D_A(fakeB)
        D_A_fake_loss = MSE_loss(D_A_fake, Variable(torch.zeros(D_A_fake.size()).cuda()))

        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        D_A_loss.backward()
        D_A_optimizer.step()
        
        # train discriminator D_B
        D_B_optimizer.zero_grad()

        D_B_real = D_B(realA)
        D_B_real_loss = MSE_loss(D_B_real, Variable(torch.ones(D_B_real.size()).cuda()))

        # fakeA = fakeA_store.query(fakeA.data)
        fakeA = fakeA_store.query(fakeA)
        D_B_fake = D_B(fakeA)
        D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))

        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        D_B_loss.backward()
        D_B_optimizer.step()
        
        # print results
        print('At epoch {}, step {} loss_D_A: {:.4f}, loss_D_B: {:.4f}, loss_G_A:{:.4f}, loss_G_B:{:.4f}\n'.format(epoch, i, D_A_loss, D_B_loss, A_cycle_loss, B_cycle_loss))
        if i>0 and (i+1) % 2000 == 0:
                torch.save(G_A.state_dict(), './model/cycleGA01_'+str(epoch)+'.ckpt')
                torch.save(G_B.state_dict(), './model/cycleGB01_'+str(epoch)+'.ckpt')

