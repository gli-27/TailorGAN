# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:33:43 2019

@author: x_j_t
"""

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
from param import get_params
from scipy.io import loadmat
import numpy as np
from networks3 import define_G2,define_C
from data_gen2 import FashionDataset
from torch.utils.data import DataLoader
from scipy.misc import imsave
#os.environ['CUDA_VISIBLE_DEVICES']='1'
saveRoot='./results/'
#torch.cuda.set_device(1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params=get_params()
batch_size=1 #params['batch_size']

gen=define_G2(norm='instance')
gen.load_state_dict(torch.load('./model/gen_adv_class_6.ckpt',map_location=device))

#classifier=define_C()
#classifier.load_state_dict(torch.load('./DATA/classifier_trans_79.ckpt',map_location=device))

#for p in classifier.parameters():
#        p.requires_grad=False
        
dataset=FashionDataset(params)
loader=DataLoader(dataset,batch_size=batch_size)

cnt=0
prev_gray, prev_src=None, None
for i,data in enumerate(loader):
    if i==0:
        prev_gray,prev_src,_,_=data
        continue
    gray_img,src_img,_,y =data
    #recon_imgs,attn,color=gen(cropped_img.to(device),part_edge.to(device))
    #switch_imgs,attn2,color2=gen(cropped_img.to(device),prev_edge.to(device))
    #_,cur_feat=classifier(gray_img.to(device))
    #_,prev_feat=classifier(prev_gray.to(device))
    recon_imgs =gen(src_img.to(device),gray_img.to(device))
    switch_imgs=gen(src_img.to(device),prev_gray.to(device))
    for i in range(batch_size):
        part_img=prev_gray[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        gen_img =recon_imgs[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        swt_img =switch_imgs[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        orig_img=y[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #mask= attn[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #mask2= attn2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #mask=np.repeat(mask, 3, axis=2)
        #print(mask.shape)
        #mask2=np.repeat(mask2, 3, axis=2)
        #colorRes=color[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #colorRes2=color2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        imsave(saveRoot+str(cnt)+'_orig.jpg', orig_img)
        imsave(saveRoot+str(cnt)+'_pred.jpg', gen_img)
        imsave(saveRoot+str(cnt)+'_switch.jpg', swt_img)
        imsave(saveRoot+str(cnt)+'_part.jpg', part_img)
        #imsave('./results/'+str(cnt)+'_color.jpg',colorRes)
        #imsave('./results/'+str(cnt)+'_mask.jpg',mask)
        #imsave('./results/'+str(cnt)+'_color2.jpg',colorRes2)
        #imsave('./results/'+str(cnt)+'_mask2.jpg',mask2)
        cnt+=1
    prev_gray=gray_img
    if cnt>=100:
        break
