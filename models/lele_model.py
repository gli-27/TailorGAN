

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import torchvision.models as models


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




class Generator(nn.Module):
    def __init__(self,input_nc = 3, output_nc = 3,ngf = 64, use_dropout=True, use_bias=False,norm_layer=nn.BatchNorm2d,n_blocks = 9,padding_type='zero'):
        super(Generator,self).__init__()
        dtype            = torch.FloatTensor


        self.fusion = nn.Sequential(
            nn.ReflectionPad2d(3),
            
            conv2d(1024,512,3,1,1),
            )

            
         
        model = []
        n_downsampling = 4
        mult = 2**(n_downsampling -1  )
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling ):
            mult = 2**(n_downsampling-i-1 ) 
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            if i == n_downsampling-3:
                self.generator1 = nn.Sequential(*model)
                model = []

        self.base = nn.Sequential(*model)
        model = []
        model += [nn.Conv2d(ngf/2, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.generator_color = nn.Sequential(*model)

        model = []
        model += [nn.Conv2d(ngf/2, 1, kernel_size=7, padding=3)]
        model += [nn.Sigmoid()]


        
    def forward(self,feature_1, feature_2, image  ):

        feature =  torch.cat(feature_1, feature_2, 1)
        feature1 = self.fusion(feature)
        feature2 = self.base(feature1)
        
        color = self.generator_color(basse)
        att = self.generator_attention(base)
            
        output = att * color + (1 - att ) * image
        return output, att, color


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.image_encoder_dis = nn.Sequential(

            conv2d(3,64,3,2, 1,normalizer=None),

            conv2d(64, 128, 3, 2, 1),

            conv2d(128, 256, 3, 2, 1),

            conv2d(256, 512, 3, 2, 1),
            )


        self.img_fc = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(True),
            )

        self.decision = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(True), 
            nn.Linear(512,1),
            nn.Sigmoid()
            )

    def forward(self, image):
        feature = self.image_encoder_dis(image)

        feature = feature.view(feature.shape[0], -1) 

        decision =  self.decision(feature)

        return decision.view(decision.size(0))


