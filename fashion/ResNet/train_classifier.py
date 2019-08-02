import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from networks2 import define_C
import networks2
from data_gen2 import FashionDataset
from param import get_params

os.environ['CUDA_VISIBLE_DEVICES']='3'
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs=80
    params=get_params()
    batch_size=params['batch_size']
    lr=1e-4
    classifier=define_C()
    optimizer=torch.optim.Adam(classifier.parameters(),lr=lr)
    LossFcn=nn.CrossEntropyLoss()
    dataset=FashionDataset(params)
    loader=DataLoader(dataset,batch_size=batch_size)
    total_step=len(loader)

    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            gray_img, src_img, collar_type, y= data
            collar_type=collar_type.to(device)
            y=y.to(device)
            pred_class=classifier(y)
            loss=LossFcn(pred_class, collar_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1,total_step,  loss.item()))
            if i>0 and (i+1) % 2000 == 0:
                torch.save(classifier.state_dict(), './model/class_colorImg_transOnly_'+str(epoch)+'.ckpt')

if __name__=="__main__":
    main()

