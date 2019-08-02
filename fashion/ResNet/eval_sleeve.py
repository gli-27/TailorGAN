

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
from sleeveTestDataGen import testFashionDataset
from torch.utils.data import DataLoader
from scipy.misc import imsave
#os.environ['CUDA_VISIBLE_DEVICES']='1'
saveRoot='./results/genSleeve10/'
#torch.cuda.set_device(1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params=get_params()
batch_size=1 #params['batch_size']

gen=define_G2(norm='instance')
gen.load_state_dict(torch.load('./trainedModel/gen_sleeve_VGG_40.ckpt'))
#classifier=define_C()
#classifier.load_state_dict(torch.load('./DATA/classifier_70.ckpt',map_location=device))
        
dataset=testFashionDataset(params,1,0)
loader=DataLoader(dataset,batch_size=batch_size)

cnt=0
classError, cbpdScore=0,0
LossFcn=nn.CrossEntropyLoss()

prev_edge, prev_src, prev_orig=None, None,None
for i,data in enumerate(loader):
    y1, y2, src1, src2,edge1,edge2, p2p1,p2p2, sleeve_type= data
    recon_imgs =gen(src1.to(device),edge1.to(device))
    switch_imgs=gen(src1.to(device),edge2.to(device))
    #pred_class=classifier(switch_imgs)
    #classError +=LossFcn(pred_class, collar_type.to(device)).item()
    
    for i in range(batch_size):
        #edge_img=prev_edge[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        gen_img =recon_imgs[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        swt_img =switch_imgs[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        orig_img=y1[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        prev_y=y2[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        s_img  =src1[i,:,:,:].permute(1,2,0).cpu().detach().numpy()
        #cbpdScore +=cpbd.compute(gen_img)
        
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
        imsave(saveRoot+str(cnt)+'_prevOirg.jpg', prev_y)
        imsave(saveRoot+str(cnt)+'_src.jpg', s_img)
        #imsave('./results/'+str(cnt)+'_color.jpg',colorRes)
        #imsave('./results/'+str(cnt)+'_mask.jpg',mask)
        #imsave('./results/'+str(cnt)+'_color2.jpg',colorRes2)
        #imsave('./results/'+str(cnt)+'_mask2.jpg',mask2)
        cnt+=1
    #prev_edge,prev_src,prev_orig=edge_img,src_img, y  
    #print(classError)
    if cnt>=500:
        break