# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:42:38 2019

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
from torch.autograd import Variable
from networks2 import vggLoss
from networks2 import define_G, define_D
from networks2 import define_C
from data_gen2 import FashionDataset
from param import get_params

os.environ['CUDA_VISIBLE_DEVICES']='0'
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs=80
    params=get_params()
    batch_size=params['batch_size']
    lr=1e-4
    #body=bodyEncoder(BasicBlock,[2,2,2,2,2])
    #part=partEncoder(BasicBlock,[2,2,2,2,2])
    gen= define_G(norm='instance')
    gen.load_state_dict(torch.load('./trainedModel/gen_vgg_inst_trans_40.ckpt',map_location=device))
    disc=define_D()
    classifier1=define_C()
    classifier2=define_C()
    classifier1.load_state_dict(torch.load('./DATA/class_colorImg_transOnly_79.ckpt',map_location=device))
    classifier2.load_state_dict(torch.load('./DATA/classifier_trans_79.ckpt',map_location=device))
    for p in classifier1.parameters():
                p.requires_grad=False
    for p in classifier2.parameters():
                p.requires_grad=False
 
    optimizer_G=torch.optim.Adam(gen.parameters(), lr=lr)
    optimizer_D=torch.optim.Adam(disc.parameters(), lr=3*lr)
    
    #L1Loss=nn.L1Loss()
    adv_loss=nn.MSELoss()
    #class_loss=nn.MSELoss() #nn.CrossEntropyLoss()
    class_loss=nn.CrossEntropyLoss()
    
    
    dataset=FashionDataset(params)
    loader=DataLoader(dataset, batch_size=batch_size)
    total_step=len(loader)
    
    
    Tensor = torch.cuda.FloatTensor
    
    #contentLoss=vggLoss()
    #contentLoss.initialize(nn.MSELoss())
    for epoch in range(num_epochs):
        prev_gray, prev_y, prev_type=None,None,None
        for i, data in enumerate(loader):
            gray_img, src_img, collar_type, y= data
            if i<1:
                prev_gray,prev_y,prev_type=gray_img,y,collar_type
                continue
            #prev_gray=prev_gray.to(device)
            #prev_y=prev_y.to(device)
            #gray_img=gray_img.to(device)
            #src_img=src_img.to(device)
            #collar_type=collar_type.to(device)
            #y=y.to(device)
            valid = Variable(Tensor(y.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(y.shape[0], 1).fill_(0.0), requires_grad=False)
            # train discriminator 
            _, class_feat=classifier2(prev_gray.to(device))
            fake_img=gen(src_img.to(device),class_feat)
            for p in disc.parameters():
                p.requires_grad=True
            optimizer_D.zero_grad()
            true_valid=disc(y.to(device).detach())
            pred_valid=disc(fake_img.to(device).detach())
            d_loss=0.5*(adv_loss(true_valid,valid)+adv_loss(pred_valid,fake))
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
            # train generator
            for p in disc.parameters():
                p.requires_grad=False
            optimizer_G.zero_grad()
            G_adv_loss=adv_loss(pred_valid, valid)
            #G_content_loss=contentLoss.get_loss(fake_img,y.float())
            ## classifying loss
            pred_class, _ =classifier1(fake_img)
            #true_class, _ =classifier1(prev_y.to(device))
            G_class_loss=class_loss(pred_class, prev_type.to(device))
            g_loss=G_adv_loss+G_class_loss
            g_loss.backward()
            optimizer_G.step()
            
            prev_gray,prev_y, prev_type=gray_img,y,collar_type
             
            print ('Epoch [{}/{}], Step [{}/{}], GLoss: {:.4f},G_adv: {:.4f}, G_class: {:.4f}, DLoss:{:.4f}' .format(epoch+1, num_epochs, i+1,total_step,  g_loss.item(),G_adv_loss.item(), G_class_loss.item(),d_loss.item()))
            if i>0 and (i+1) % 2200==0:
                torch.save(gen.state_dict(), './model/gen_adv_class_'+str(epoch)+'.ckpt')
                torch.save(disc.state_dict(), './model/disc_'+str(epoch)+'.ckpt')
                break
    
if __name__=="__main__":
    main()

