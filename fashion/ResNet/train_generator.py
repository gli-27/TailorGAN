# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:57:16 2019

@author: x_j_t
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from networks3 import define_G2, define_C
from networks3 import vggLoss 
#from networks import bodyEncoder, partEncoder
from data_gen2 import FashionDataset
from param import get_params

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs=40
    params=get_params()
    batch_size=params['batch_size']
    lr=1e-4
    #body=bodyEncoder(BasicBlock,[2,2,2,2,2])
    #part=partEncoder(BasicBlock,[2,2,2,2,2])
    gen= define_G2(norm='instance')
    #gen.load_state_dict(torch.load('./model/gen_vgg_inst_trans_bflip_12.ckpt',map_location=device))
    #classifier=define_C()
    #classifier.load_state_dict(torch.load('./DATA/class_trans_colorJitter_79.ckpt',map_location=device))
    #for p in classifier.parameters():
    #    p.requires_grad=False

    optimizer=torch.optim.Adam(gen.parameters(), lr=lr)
    L1Loss=nn.L1Loss()
    dataset=FashionDataset(params)
    loader=DataLoader(dataset, batch_size=batch_size)
    total_step=len(loader)
    
    contentLoss=vggLoss()
    contentLoss.initialize(nn.MSELoss())
    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            gray_img, src_img, _, y= data
            gray_img=gray_img.to(device)
            src_img=src_img.to(device)
            y=y.to(device)

            #_, class_feat=classifier(gray_img)
            outputs=gen(src_img, gray_img)
            #loss=L1Loss(outputs, y.float())
            loss=contentLoss.get_loss(outputs,y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1,total_step,  loss.item()))
            if i>0 and (i+1) % 2000 == 0:
                torch.save(gen.state_dict(), './model/gen_gray_VGG_'+str(epoch+1)+'.ckpt')

    
if __name__=="__main__":
    main()
