import os
import sys
import cycleNet
from testDataGen import testFashionDataset
import p2pNetworks
import torch.nn as nn
from networks2 import define_C
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import itertools
import random
from param import get_params
from scipy.misc import imsave

#os.environ['CUDA_VISIBLE_DEVICES']='3'

G= p2pNetworks.generator(64)
G.load_state_dict(torch.load('./trainedModel/p2pG_49.ckpt',map_location={'cuda':'cpu'}))
G.cuda()

classifier=define_C()
classifier.load_state_dict(torch.load('./DATA/classifier_70.ckpt',map_location={'cuda':'cpu'}))
LossFcn=nn.CrossEntropyLoss()
for p in classifier.parameters():
    p.requires_grad=False
    
params=get_params()
batch_size= 1 #params['batch_size']

dataset=testFashionDataset(params,5,1)
loader=DataLoader(dataset, batch_size=batch_size)

cnt=0

class_error=0

for i, data in enumerate(loader):
    y1, y2, src1, src2, _, _, part1,part2,edge1,edge2, p2p1,p2p2, collar_type= data
        
    p2p1, p2p2= Variable(p2p1.cuda()),Variable(p2p2.cuda())
    recon_res=G(p2p1)
    switch_res=G(p2p2)
    #pred_class=classifier(switch_res)
    #class_error+=LossFcn(pred_class, collar_type.cuda())
    print(class_error)
    for j in range(batch_size):
        rec_img =recon_res[j,:,:,:].permute(1,2,0).cpu().detach().numpy()
        swt_img =switch_res[j,:,:,:].permute(1,2,0).cpu().detach().numpy()
        edge_img =edge2[j,:,:,:].permute(1,2,0).cpu().detach().numpy()
        orig_img=y1[j,:,:,:].permute(1,2,0).cpu().detach().numpy()
        imsave('./results/p2p51/'+str(cnt)+'_p2p_05_orig.jpg', orig_img)
        #imsave('./results/p2p_05/'+str(cnt)+'_p2p_01_pred.jpg', rec_img)
        #imsave('./results/p2p_05/'+str(cnt)+'_p2p_01_part.jpg', edge_img)
        imsave('./results/p2p51/'+str(cnt)+'_p2p_05_switch.jpg', swt_img)
    cnt+=1
    if cnt>=400:
        print('reach the limit quit\n')
        break

