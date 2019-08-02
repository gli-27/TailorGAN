# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:49:30 2019

@author: x_j_t
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

class testFashionDataset(Dataset):
    def __init__(self,params, type1=0, type2=1):
        info_file_path= './DATA/train_sleeve.csv'
        if not os.path.isfile(info_file_path):
            print('train file is not found, exit on error\n')
            sys.exit(1)
        self.data_df=pd.read_csv(info_file_path)
        if type1<0 or type1>11 or type2<0 or type2>11:
            print('collar type input is invalid, can only be an integer number between [0,11]\n')
            sys.exit(1)
        
        self.t1_data_df=self.data_df.loc[self.data_df.type==type1]
        self.t2_data_df=self.data_df.loc[self.data_df.type==type2]
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
        self.masktransform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128,128),interpolation=2),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomAffine(degrees=5,scale=(0.85,1.15),translate=(0.1,0.05)),
                    #transforms.ColorJitter(brightness=0,contrast=0.5,saturation=0.5),
                    transforms.ToTensor(),
                ])
        self.p2pTransform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256,256),interpolation=2),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),
                    #transforms.ColorJitter(brightness=0,contrast=0.8,saturation=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5,0.5],std=[0.5,0.5,0.5,0.5])
                    ])

    def __getitem__(self,index):
        #imgH,imgW=self.params['IMG_HEIGHT'],self.params['IMG_WIDTH']
        #imgInd1=np.random.choice(len(self.t1_data_df),1)[0]
        #imgInd2=np.random.choice(len(self.t2_data_df),1)[0]
        row_df1=self.t1_data_df.iloc[index]
        row_df2=self.t2_data_df.iloc[index]
        
        img_name1=row_df1.imageName
        orig_img1=io.imread(os.path.join('./DATA/Sleeve/target/',img_name1))
        img_name2=row_df2.imageName
        orig_img2=io.imread(os.path.join('./DATA/Sleeve/target/',img_name2))
        
        #edge_imgPath1=row_df1.edge_imgPath
        edge_img1 =io.imread(os.path.join('./DATA/Sleeve/edge',img_name1))
        edge_img1=np.repeat(edge_img1[:,:,np.newaxis], 3, axis=2)
        #edge_imgPath2=row_df2.edge_imgPath
        edge_img2 =io.imread(os.path.join('./DATA/Sleeve/edge',img_name2))
        edge_img2=np.repeat(edge_img2[:,:,np.newaxis], 3, axis=2)
        
        gray_img1 = 0.2125*orig_img1[:,:,0] + 0.7154*orig_img1[:,:,1] + 0.0721*orig_img1[:,:,2]
        gray_img2 = 0.2125*orig_img2[:,:,0] + 0.7154*orig_img2[:,:,1] + 0.0721*orig_img2[:,:,2]
        
        """
        #gray_img1= np.repeat(gray_img1[:, :, np.newaxis], 3, axis=2)
        #gray_img2= np.repeat(gray_img2[:, :, np.newaxis], 3, axis=2)

        #mask1=np.zeros((256,256,3))
        #mask2=np.zeros((256,256,3))

        #bbox_x1,bbox_y1=row_df1.landmark1_x,row_df1.landmark1_y
        #bbox_x2,bbox_y2=row_df1.landmark2_x,row_df1.landmark2_y
        #centX,centY=int((bbox_x1+bbox_x2)/2.0), int((bbox_y1+bbox_y2)/2.0)
        #tY2,bY2=max(centY-40,0),min(256,centY+40)
        #tX2,bX2=max(centX-40,0),min(256,centX+40)
        #tY,tX=max(centY-64,0),max(0,centX-64)
        #bY,bX=min(tY+128,256),min(tX+128,256)
        #tY,tX=max(bY-128,0),max(0,bX-128)
        #mask1[tY2:bY2,tX2:bX2,:]=1.0
        #part_edge_img1=edge_img1[tY:bY,tX:bX,:].copy()
        #src_img1=np.multiply(1.0-mask1,orig_img1)
        

        bbox_x1,bbox_y1=row_df2.landmark1_x,row_df2.landmark1_y
        bbox_x2,bbox_y2=row_df2.landmark2_x,row_df2.landmark2_y
        centX,centY=int((bbox_x1+bbox_x2)/2.0), int((bbox_y1+bbox_y2)/2.0)
        tY2,bY2=max(centY-40,0),min(256,centY+40)
        tX2,bX2=max(centX-40,0),min(256,centX+40)
        tY,tX=max(centY-64,0),max(0,centX-64)
        bY,bX=min(tY+128,256),min(tX+128,256)
        tY,tX=max(bY-128,0),max(0,bX-128)
        mask2[tY2:bY2,tX2:bX2,:]=1.0
        part_edge_img2=edge_img2[tY:bY,tX:bX,:].copy()
        src_img2=np.multiply(1.0-mask2,orig_img2)
        """
        
        src_img1=io.imread(os.path.join('./DATA/Sleeve/crop',img_name1))
        src_img2=io.imread(os.path.join('./DATA/Sleeve/crop',img_name2))
        
        p2pImg1=np.concatenate((src_img1, gray_img1[:,:,np.newaxis]), axis=2)
        p2pImg2=np.concatenate((src_img1, gray_img2[:,:,np.newaxis]), axis=2)

        orig_img_tensor1=self.transform(np.uint8(orig_img1))
        orig_img_tensor2=self.transform(np.uint8(orig_img2))
        src_img_tensor1=self.transform(src_img1.astype(np.uint8))
        src_img_tensor2=self.transform(src_img2.astype(np.uint8))
        edge_tensor1=self.transform(np.uint8(edge_img1))
        edge_tensor2=self.transform(np.uint8(edge_img2))
       
        p2p_tensor1 = self.p2pTransform(np.uint8(p2pImg1))
        p2p_tensor2 = self.p2pTransform(np.uint8(p2pImg2))
       
        collar_type= row_df2.type
        
        return [orig_img_tensor1,orig_img_tensor2, src_img_tensor1,src_img_tensor2, edge_tensor1,edge_tensor2, p2p_tensor1, p2p_tensor2, collar_type]
    
    def __len__(self):
        return 200 #len(self.data_df)
    


if __name__=='__main__':
    params=param.get_params()
    dataset=testFashionDataset(params,0,1)
    loader=DataLoader(dataset, batch_size=1)

    
    for i, data in enumerate(loader):
        if i>0:
            break
            #print("i={}\n".format(i))
        y1, y2, src1, src2, part1,part2,p2p1,p2p2= data
            
        tgt1 =y1[0,:,:,:].permute(1,2,0).numpy()
        tgt2 =y2[0,:,:,:].permute(1,2,0).numpy()
            
        tgt1 =(tgt1*0.5 +0.5)*255.0
        tgt2 =(tgt2*0.5 +0.5)*255.0
            
        body1 =src1[0,:,:,:].permute(1,2,0).numpy()
        body2 =src2[0,:,:,:].permute(1,2,0).numpy()
            
        body1 =(body1*0.5 +0.5)*255.0
        body2 =(body2*0.5 +0.5)*255.0
        
        pe1 =part1[0,:,:,:].permute(1,2,0).numpy()
        pe2 =part2[0,:,:,:].permute(1,2,0).numpy()
            
        pe1 =(pe1*0.5 +0.5)*255.0
        pe2 =(pe2*0.5 +0.5)*255.0
        
        
        
        f=plt.figure(i+10)
        plt.imshow(pe1.astype(np.uint8))
        f.show()
            
        f=plt.figure(i)
        plt.imshow(pe2.astype(np.uint8))
        f.show()
