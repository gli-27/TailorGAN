#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:07:48 2019

@author: justintian
"""

import sys
import os
import numpy as np
import torch
from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import param

class FashionDataset(Dataset):
    def __init__(self,params):
        info_file_path='./DATA/leave_5_out.csv' #params['train_anno_path']
        if not os.path.isfile(info_file_path):
            print('train file is not found, exit on error\n')
            sys.exit(1)
        self.data_df=pd.read_csv(info_file_path)
        self.params=params
        self.transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128,128),interpolation=2),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.8,contrast=0.8,saturation=0.8),
                    transforms.RandomAffine(degrees=15,scale=(0.85,1.15),translate=(0.10,0.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                    ])
        self.orig_transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128,128),interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
    
    def __getitem__(self,index):
        #imgH,imgW=self.params['IMG_HEIGHT'],self.params['IMG_WIDTH']
        #type1=np.random.choice(12,1)[0]
        #self.t1_data_df=self.data_df.loc[self.data_df.collar_type==type1]
        imgInd=np.random.choice(len(self.data_df),1)[0]
        row_df=self.data_df.iloc[imgInd]
        orig_imgPath=row_df.tgt_imgPath
        edge_imgPath=row_df.edge_imgPath

        orig_img=io.imread(orig_imgPath)
        edge_img=io.imread(edge_imgPath)

        gray_img= 0.2125*orig_img[:,:,0] + 0.7154*orig_img[:,:,1] + 0.0721*orig_img[:,:,2] #255.0*rgb2gray(orig_img)
        #randVal=np.random.uniform()
        #if randVal>0.5:
        #    gray_img=255.0-gray_img
        gray_img= np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
        #print(np.max(gray_img))
        mask=np.zeros((256,256,3))
        bbox_x1,bbox_y1=row_df.landmark1_x,row_df.landmark1_y
        bbox_x2,bbox_y2=row_df.landmark2_x,row_df.landmark2_y
        centX,centY=int((bbox_x1+bbox_x2)/2.0), int((bbox_y1+bbox_y2)/2.0) 
        tY2,bY2=max(centY-40,0),min(256,centY+40)
        tX2,bX2=max(centX-40,0),min(256,centX+40)
        tY,tX=max(centY-64,0),max(0,centX-64)
        bY,bX=min(tY+128,256),min(tX+128,256)
        tY,tX=max(bY-128,0),max(0,bX-128)
        mask[tY2:bY2,tX2:bX2,:]=1.0

        part_edge_img=edge_img[tY:bY,tX:bX,:].copy()
        #print(orig_img.shape)
        #blur_img=255.0*filters.gaussian(orig_img, 8)
        #print(np.max(blur_img))
        src_img=np.multiply(1.0-mask,orig_img)#np.multiply(mask,blur_img)
        
        collar_type= row_df.collar_type
        src_img_tensor=self.orig_transform(src_img.astype(np.uint8))
        orig_img_tensor=self.orig_transform(np.uint8(orig_img))
        gray_img_tensor=self.orig_transform(np.uint8(gray_img))
        edge_img_tensor=self.orig_transform(np.uint8(edge_img))
        part_edge_tensor=self.transform(np.uint8(part_edge_img))
        return [part_edge_tensor, src_img_tensor, collar_type, orig_img_tensor]
    
    def __len__(self):
        return 5000 #len(self.data_df)
    


if __name__=='__main__':
    params=param.get_params()
    dataset=FashionDataset(params)
    loader=DataLoader(dataset, batch_size=4)

    for epoch in range(1):
        #print("epoch = {}\n".format(epoch))
        for i, data in enumerate(loader):
            if i>0:
                break
            #print("i={}\n".format(i))
            gray, blurred, y= data
            #print(part_edge.size(),cropped_img.size())
            gImg=gray[0,:,:,:].permute(1,2,0).numpy()
            blur=blurred[0,:,:,:].permute(1,2,0).numpy()
            #body=cropped_img[0,:,:,:].permute(1,2,0).numpy()
            #mask=part_mask[0,:,:,:].numpy()
            tgt =y[0,:,:,:].permute(1,2,0).numpy()
            #print(cropped_img.shape)
            blur=(blur*0.5+0.5)*255.0
            gImg=(gImg*0.5+0.5)*255.0
            #body=(body*0.5+0.5)*255.0 #body*255.0
            tgt =(tgt*0.5 +0.5)*255.0
            #test=(test*0.5+ 0.5)*255.0
            #part=np.swapaxes(part,0,2)
            #body=np.swapaxes(body,0,2)
            #print(np.max(mask),np.min(mask))
            f=plt.figure(i)
            plt.imshow(blur.astype(np.uint8))
            f.show()
            
            f=plt.figure(i+10)
            plt.imshow(tgt.astype(np.uint8))
            f.show()
            
            f=plt.figure(i+20)
            plt.imshow(gImg.astype(np.uint8))
            f.show()
            #f=plt.figure(i+10)
        
    
    
    
    
    
    
    
    
    
    
    
