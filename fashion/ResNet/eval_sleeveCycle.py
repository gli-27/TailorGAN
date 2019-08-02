# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:13:33 2019

@author: x_j_t
"""

import torch
from torchvision import transforms
import sys
import os
from param import get_params
from scipy.io import loadmat
import numpy as np
import cycleNet
from networks3 import define_G2,define_C
import torch.nn as nn
from sleeveTestDataGen import testFashionDataset
from torch.utils.data import DataLoader
from scipy.misc import imsave
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#torch.cuda.set_device(1)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params=get_params()
batch_size=1 #params['batch_size']

gen=cycleNet.generator(3,3,32,9)

gen.load_state_dict(torch.load('./trainedModel/cycleSleeveGA_35.ckpt',map_location={'cuda':'cpu'}))
gen.cuda()

#classifier=define_C()
#classifier.load_state_dict(torch.load('./DATA/classifier_70.ckpt',map_location=device))
#classError, cbpdScore=0,0
#LossFcn=nn.CrossEntropyLoss()
#for p in classifier.parameters():
#    p.requires_grad=False

dataset=testFashionDataset(params,1,0)
loader=DataLoader(dataset,batch_size=batch_size)

cnt=0

for i,data in enumerate(loader):
    
    y1, y2, src1, src2, edge1,edge2, p2p1,p2p2, collar_type= data
    #recon_imgs,attn,color=gen(cropped_img.to(device),part_edge.to(device))
    #switch_imgs,attn2,color2=gen(cropped_img.to(device),prev_edge.to(device))
    switch_imgs=gen(y1.to(device))
    #pred_class=classifier(switch_imgs)
    #classError +=LossFcn(pred_class, collar_type.to(device)).item()

    for i in range(batch_size):
        swt_img =switch_imgs[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        orig_img=y1[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        imsave('./results/sleeveCycle10/'+str(cnt)+'_orig.jpg', orig_img)
        imsave('./results/sleeveCycle10/'+str(cnt)+'_switch.jpg', swt_img)
        cnt+=1
    #print(classError)
    if cnt>=400:
        break
