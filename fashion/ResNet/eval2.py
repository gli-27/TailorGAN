#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:08:24 2019

@author: justintian
"""

import torch
from torchvision import transforms
import sys
import os
import cpbd
from param import get_params
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
from networks3 import define_G2,define_C
from testDataGen import testFashionDataset
from torch.utils.data import DataLoader
from scipy.misc import imsave
#os.environ['CUDA_VISIBLE_DEVICES']='1'
saveRoot='./results/oneout/'
#torch.cuda.set_device(1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params=get_params()
batch_size=1 #params['batch_size']

gen=define_G2(norm='instance')
#gen.load_state_dict(torch.load('./trainedModel/gen_out5_8.ckpt',map_location=device))
gen.load_state_dict(torch.load('./trainedModel/gen_2b_disc_class_5.ckpt',map_location=device))


classifier=define_C()
classifier.load_state_dict(torch.load('./DATA/classifier_70.ckpt',map_location=device))
        
dataset=testFashionDataset(params,1,5)
loader=DataLoader(dataset,batch_size=batch_size)

cnt=0
classError, cbpdScore=0,0
LossFcn=nn.CrossEntropyLoss()
for p in classifier.parameters():
    p.requires_grad=False
prev_gray, prev_src=None, None
for i,data in enumerate(loader):
    #if i==0:
    #    prev_gray,prev_src,_,_=data
    #    continue
    y1, y2, src1, src2, mask1,mask2, part1,part2,edge1,edge2, p2p1,p2p2, collar_type= data
    recon_imgs =gen(src1.to(device),part1.to(device))
    switch_imgs=gen(src1.to(device),part2.to(device))
    pred_class=classifier(switch_imgs)
    classError +=LossFcn(pred_class, collar_type.to(device)).item()
    
    for i in range(batch_size):
        edge_img=edge2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        prev_orig =y2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        swt_img =switch_imgs[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        orig_img=y1[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        mask1_img=mask1[i,:,:,:].permute(1,2,0).numpy()
        part_eval=np.multiply(mask1_img,swt_img)
        #cbpdScore +=cpbd.compute(gen_img)
        
        #mask= attn[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #mask2= attn2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #mask=np.repeat(mask, 3, axis=2)
        #print(mask.shape)
        #mask2=np.repeat(mask2, 3, axis=2)
        #colorRes=color[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #colorRes2=color2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        imsave(saveRoot+str(cnt)+'_15_orig.jpg', orig_img)
        imsave(saveRoot+str(cnt)+'_15_prevOrig.jpg', prev_orig)
        imsave(saveRoot+str(cnt)+'_15_switch.jpg', swt_img)
        imsave(saveRoot+str(cnt)+'_15_edge.jpg', edge_img)
        imsave(saveRoot+str(cnt)+'_15_partEval.jpg',part_eval)
        #imsave('./results/'+str(cnt)+'_color.jpg',colorRes)
        #imsave('./results/'+str(cnt)+'_mask.jpg',mask)
        #imsave('./results/'+str(cnt)+'_color2.jpg',colorRes2)
        #imsave('./results/'+str(cnt)+'_mask2.jpg',mask2)
        cnt+=1
    print(classError)
    if cnt>=100:
        break
