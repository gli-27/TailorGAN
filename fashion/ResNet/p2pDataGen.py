#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:53:26 2019

@author: justintian
"""

import sys
import os
import numpy as np
import torch
import random
from skimage import io
from skimage import filters
from skimage import transform as stf
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import param

class p2pFashionDataset(Dataset):
    def __init__(self,params):
        info_file_path=params['train_anno_path']
        if not os.path.isfile(info_file_path):
            print('train file is not found, exit on error\n')
            sys.exit(1)
        self.data_df=pd.read_csv(info_file_path)
        self.params=params
        self.transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256,256),interpolation=2),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),
                    #transforms.ColorJitter(brightness=0,contrast=0.8,saturation=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5,0.5],std=[0.5,0.5,0.5,0.5])
                ])
        self.orig_transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256),interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
    
    def __getitem__(self,index):
        #imgH,imgW=self.params['IMG_HEIGHT'],self.params['IMG_WIDTH']
        imgInd=np.random.choice(len(self.data_df),1)[0]
        row_df=self.data_df.iloc[imgInd]
        orig_imgPath=row_df.tgt_imgPath
        orig_img=io.imread(orig_imgPath)
        #gray_img= 255.0*rgb2gray(orig_img)
        gray_img=0.2125*(orig_img[:,:,0]) + 0.7154*(orig_img[:,:,1])+ 0.0721*(orig_img[:,:,2])
        rScale=np.random.uniform(0.85,1.15)
        rAng  =np.random.uniform(0,0.2)-0.1
        rHTrans=int((np.random.uniform(0,0.3)-0.15)*256)
        rVTrans=int((np.random.uniform(0,0.05)-0.025)*256)
        g_trans=stf.AffineTransform(scale=(rScale,rScale),rotation=rAng, translation=(rHTrans,rVTrans))
        g_warp =stf.warp(gray_img,g_trans)
        
        #gray_img= np.repeat(g_warp[:, :, np.newaxis], 3, axis=2)
        #if randVal>0.5:
        #    gray_img=255.0-gray_img
        #gray_img= np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
        #print(np.max(gray_img))
        bbox_x1,bbox_y1=row_df.landmark1_x,row_df.landmark1_y
        bbox_x2,bbox_y2=row_df.landmark2_x,row_df.landmark2_y
        centX,centY=int((bbox_x1+bbox_x2)/2.0), int((bbox_y1+bbox_y2)/2.0)
        mask=np.zeros((256,256,3))
        #mask_path=row_df.mask_imgPath
        #mask =io.imread(mask_path)
        tY,bY=max(centY-50,0),min(256,centY+50)
        lX,rX=max(centX-60,0),min(256,centX+60)
        mask[tY:bY,lX:rX,:]=1.0
        #print(orig_img.shape)
        #blur_img=255.0*filters.gaussian(orig_img, 8)
        #print(np.max(blur_img))
        src_img=np.multiply(1.0-mask,orig_img)#np.multiply(mask,blur_img)
        src_img=np.concatenate((src_img, g_warp[:,:,np.newaxis]), axis=2)
        #print(src_img.shape, g_warp.shape)
        #collar_type= row_df.collar_type
        src_img_tensor=self.transform(src_img.astype(np.uint8))
        orig_img_tensor=self.orig_transform(np.uint8(orig_img))
        
        return [src_img_tensor,orig_img_tensor]
    
    def __len__(self):
        return len(self.data_df)
    
class testFashionDataset(Dataset):
    def __init__(self,params):
        info_file_path=params['train_anno_path']
        if not os.path.isfile(info_file_path):
            print('train file is not found, exit on error\n')
            sys.exit(1)
        self.data_df=pd.read_csv(info_file_path)
        self.params=params
        self.transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256,256),interpolation=2),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),
                    #transforms.ColorJitter(brightness=0,contrast=0.8,saturation=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5,0.5],std=[0.5,0.5,0.5,0.5])
                ])
        self.orig_transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256),interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])

    def __getitem__(self,index):
        #imgH,imgW=self.params['IMG_HEIGHT'],self.params['IMG_WIDTH']
        imgInd=np.random.choice(len(self.data_df),1)[0]
        row_df=self.data_df.iloc[imgInd]
        orig_imgPath=row_df.tgt_imgPath
        orig_img=io.imread(orig_imgPath)
        #gray_img= 255.0*rgb2gray(orig_img)
        gray_img=0.2125*(orig_img[:,:,0]) + 0.7154*(orig_img[:,:,1])+ 0.0721*(orig_img[:,:,2])
        rScale=np.random.uniform(0.85,1.15)
        rAng  =np.random.uniform(0,0.2)-0.1
        rHTrans=int((np.random.uniform(0,0.3)-0.15)*256)
        rVTrans=int((np.random.uniform(0,0.05)-0.025)*256)
        g_trans=stf.AffineTransform(scale=(rScale,rScale),rotation=rAng, translation=(rHTrans,rVTrans))
        g_warp =stf.warp(gray_img,g_trans)

        imgInd2=np.random.choice(len(self.data_df),1)[0]
        row_df2=self.data_df.iloc[imgInd2]
        orig_imgPath2=row_df2.tgt_imgPath
        orig_img2=io.imread(orig_imgPath2)
        gray_img2=0.2125*(orig_img2[:,:,0]) + 0.7154*(orig_img2[:,:,1])+ 0.0721*(orig_img2[:,:,2])
        rScale=np.random.uniform(0.85,1.15)
        rAng  =np.random.uniform(0,0.2)-0.1
        rHTrans=int((np.random.uniform(0,0.3)-0.15)*256)
        rVTrans=int((np.random.uniform(0,0.05)-0.025)*256)
        g_trans=stf.AffineTransform(scale=(rScale,rScale),rotation=rAng, translation=(rHTrans,rVTrans))
        g_warp2 =stf.warp(gray_img2,g_trans)

        #gray_img= np.repeat(g_warp[:, :, np.newaxis], 3, axis=2)
        #if randVal>0.5:
        #    gray_img=255.0-gray_img
        #gray_img= np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
        #print(np.max(gray_img))
        bbox_x1,bbox_y1=row_df.landmark1_x,row_df.landmark1_y
        bbox_x2,bbox_y2=row_df.landmark2_x,row_df.landmark2_y
        centX,centY=int((bbox_x1+bbox_x2)/2.0), int((bbox_y1+bbox_y2)/2.0)
        mask=np.zeros((256,256,3))
        #mask_path=row_df.mask_imgPath
        #mask =io.imread(mask_path)
        tY,bY=max(centY-50,0),min(256,centY+50)
        lX,rX=max(centX-60,0),min(256,centX+60)
        mask[tY:bY,lX:rX,:]=1.0
        #print(orig_img.shape)
        #blur_img=255.0*filters.gaussian(orig_img, 8)
        #print(np.max(blur_img))
        src_img=np.multiply(1.0-mask,orig_img)#np.multiply(mask,blur_img)
        x1=np.concatenate((src_img, g_warp[:,:,np.newaxis]), axis=2)
        x2=np.concatenate((src_img, g_warp2[:,:,np.newaxis]), axis=2)
        #print(src_img.shape, g_warp.shape)
        #collar_type= row_df.collar_type
        x1_tensor=self.transform(x1.astype(np.uint8))
        x2_tensor=self.transform(x2.astype(np.uint8))
        orig_img_tensor=self.orig_transform(np.uint8(orig_img))
        
        gray_img2= np.repeat(gray_img2[:, :, np.newaxis], 3, axis=2)
        gray_tensor=self.orig_transform(np.uint8(gray_img2))
        return [x1_tensor,x2_tensor,gray_tensor,orig_img_tensor]

    def __len__(self):
        return len(self.data_df)

if __name__=='__main__':
    params=param.get_params()
    dataset=p2pFashionDataset(params)
    loader=DataLoader(dataset, batch_size=4)

    for epoch in range(1):
        #print("epoch = {}\n".format(epoch))
        for i, data in enumerate(loader):
            if i>0:
                break
            #print("i={}\n".format(i))
            src_img, y= data
            """
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
            """
