#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:56:53 2019

@author: justintian
"""

import sys
import os
import numpy as np
import torch
import random
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

class CycleFashionDataset(Dataset):
    def __init__(self,params, type1=0, type2=1):
        info_file_path=params['train_anno_path']
        if not os.path.isfile(info_file_path):
            print('train file is not found, exit on error\n')
            sys.exit(1)
        self.data_df=pd.read_csv(info_file_path)
        if type1<0 or type1>11 or type2<0 or type2>11:
            print('collar type input is invalid, can only be an integer number between [0,11]\n')
            sys.exit(1)
        
        self.t1_data_df=self.data_df.loc[self.data_df.collar_type==type1]
        self.t2_data_df=self.data_df.loc[self.data_df.collar_type==type2]
        self.params=params
        self.transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128,128),interpolation=2),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomAffine(degrees=5,scale=(0.85,1.15),translate=(0.1,0.05)),
                    #transforms.ColorJitter(brightness=0,contrast=0.5,saturation=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
    
    def __getitem__(self,index):
        #imgH,imgW=self.params['IMG_HEIGHT'],self.params['IMG_WIDTH']
        imgInd1=np.random.choice(len(self.t1_data_df),1)[0]
        imgInd2=np.random.choice(len(self.t2_data_df),1)[0]
        row_df1=self.t1_data_df.iloc[imgInd1]
        row_df2=self.t2_data_df.iloc[imgInd2]
        orig_imgPath1=row_df1.tgt_imgPath
        orig_img1=io.imread(orig_imgPath1)
        orig_imgPath2=row_df2.tgt_imgPath
        orig_img2=io.imread(orig_imgPath2)
        orig_img_tensor1=self.transform(np.uint8(orig_img1))
        orig_img_tensor2=self.transform(np.uint8(orig_img2))
        return [orig_img_tensor1,orig_img_tensor2]
    
    def __len__(self):
        return len(self.data_df)
    


if __name__=='__main__':
    params=param.get_params()
    dataset=CycleFashionDataset(params,0,8)
    loader=DataLoader(dataset, batch_size=4)

    for epoch in range(1):
        #print("epoch = {}\n".format(epoch))
        for i, data in enumerate(loader):
            if i>0:
                break
            #print("i={}\n".format(i))
            y1, y2= data
            
            tgt1 =y1[0,:,:,:].permute(1,2,0).numpy()
            tgt2 =y2[0,:,:,:].permute(1,2,0).numpy()
            
            tgt1 =(tgt1*0.5 +0.5)*255.0
            tgt2 =(tgt2*0.5 +0.5)*255.0
            
            
            
            f=plt.figure(i+10)
            plt.imshow(tgt1.astype(np.uint8))
            f.show()
            
            f=plt.figure(i)
            plt.imshow(tgt2.astype(np.uint8))
            f.show()
            

