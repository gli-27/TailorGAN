# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:53:02 2018

@author: x_j_t
"""
import os
def get_params():
    param={}
    dn=1
    param['IMG_HEIGHT']=int(256/dn)
    param['IMG_WIDTH'] =int(256/dn)
    param['scale_max']=1.1
    param['scale_min']=0.9
    param['max_rotate_degree']=15
    param['max_sat_factor']=0.1
    param['max_px_shift']=20
    param['posemap_downsample']=2.0
    param['sigma_landmark']= 7/2.0
    param['n_landmarks']=2
    param['n_parts']=1
    param['parts']=[[0,1]]
    param['batch_size']=4
    param['data_root']='./DATA/Res'
    param['max_color_shift']=0.1
    param['train_anno_path']=os.path.join(param['data_root'],'trainResAnno.csv')
    param['test_anno_path'] =os.path.join(param['data_root'],'testResAnno.csv')
    param['n_training_iter']=200000
    param['model_save_dir']='./model'
    param['model_save_interval']=5000
    param["test_iter"]=500
    param['resPath']='./results'
    param['baseLayer']=15
    param['alpha1']=0.001 #0.0005
    param['alpha2']=1.00
    return param