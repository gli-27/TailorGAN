import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse
from data import data_loader
from models import lele_model
from mode import network_revised

from torch.nn import init
import utils
from options.options import Options




def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

class Trainer():
    def __init__(self, config):

        self.opt = Options().parse()


        self.generator = lele_model.Generator()
        self.discriminator = lele_model.Discriminator()

        self.edgeEncoder = network_revised.edgeEncoder()

        self.srcEncoder = network_revised.srcEncoder()




        self.bce_loss_fn = nn.BCELoss()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config
        self.ones = Variable(torch.ones(config.batch_size), requires_grad=False)
        self.zeros = Variable(torch.zeros(config.batch_size), requires_grad=False)

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            self.discriminator     = nn.DataParallel(self.discriminator, device_ids=device_ids).cuda()


            self.srcEncoder =  self.srcEncoder.cuda(device=config.cuda1)

            self.edgeEncoder = self.edgeEncoder.cuda(device=config.cuda1)




            self.bce_loss_fn   = self.bce_loss_fn.cuda(device=config.cuda1)
            self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
            self.l1_loss_fn = self.l1_loss_fn.cuda(device=config.cuda1)
            self.ones          = self.ones.cuda(device=config.cuda1)
            self.zeros          = self.zeros.cuda(device=config.cuda1)
# #########single GPU#######################

#         if config.cuda:
#             device_ids = [int(i) for i in config.device_ids.split(',')]
#             self.generator     = self.generator.cuda(device=config.cuda1)
#             self.encoder = self.encoder.cuda(device=config.cuda1)
#             self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
#             self.l1_loss_fn =  nn.L1Loss().cuda(device=config.cuda1)

        self.srcEncoder.load_state_dict(torch.load(
            './checkpoints/TailorGAN_Garmentset/path/TailorGAN_Garment_iterSyn_srcE_100.pth' 
        ))
        
        self.edgeEncoder.load_state_dict(torch.load(
            './checkpoints/TailorGAN_Garmentset/path/TailorGAN_Garment_iterSyn_edgeE_100.pth'
        ))

        for param in srcEncoder.parameters():
            param.requires_grad = False

        for param in edgeEncoder.parameters():
            param.requires_grad = False



        initialize_weights(self.generator)
        nitialize_weights(self.discriminator)
        self.start_epoch = 0
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_d = torch.optim.Adam( self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.dataset = data_loader.InterDataset(self.opt)

        self.data_loader = DataLoader(self.dataset,
                                      batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
        self.dataset_size = len(dataset)

        data_iter = iter(self.data_loader)
        data_iter.next()

    def fit(self):

        config = self.config

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
        for epoch in range(self.start_epoch, config.max_epochs):
            for step, data_batch in enumerate(self.data_loader):
                [part_edge_tensor, refer_part_edge_tensor, src_img_tensor, collar_type, orig_img_tensor] = data_batch
                t1 = time.time()

                if config.cuda:
                    part_edge_tensor = Variable(part_edge_tensor.float()).cuda(device=config.cuda1)
                    refer_part_edge_tensor = Variable(refer_part_edge_tensor.float()).cuda(device=config.cuda1)
                    src_img_tensor    = Variable(src_img_tensor.float()).cuda(device=config.cuda1)
                    collar_type = Variable(collar_type.float()).cuda(device=config.cuda1)
                    orig_img_tensor = Variable(orig_img_tensor.float()).cuda(device=config.cuda1)
               







                #train the discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad =  True


                edge_feature = self.edgeEncoder(refer_part_edge_tensor)

                image_feature = self.srcEncoder(src_img_tensor)



                fake_im, _ , _  = self.generator( edge_feature.detach(), image_feature.detach())

                D_real  = self.discriminator(orig_img_tensor)

                loss_real = self.bce_loss_fn(D_real, self.ones)

                # train with fake image
                D_fake  = self.discriminator(fake_im.detach())
                loss_fake = self.bce_loss_fn(D_fake, self.zeros)

                loss_disc = loss_real  + loss_fake 

                loss_disc.backward()
                self.opt_d.step()
                self._reset_gradients()

                #train the generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_im, att ,colors  = self.generator(edge_feature.detach(), image_feature.detach() )
                D_fake  = self.discriminator(fake_im)

                loss = self.bce_loss_fn(D_fake, self.ones)
                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()
                t2 = time.time()

                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    print("[{}/{}][{}/{}]  ,  loss_disc: {:.8f},  loss_gen: {:.8f}  , data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss_disc.data[0],loss.data[0], t1-t0,  t2 - t1))

                if (step) % (int(num_steps_per_epoch  / 20 )) == 0 :
                    atts_store = att.data.contiguous().view(config.batch_size*16,1,128,128)
                    colors_store = colors.data.contiguous().view(config.batch_size * 16,3,128,128)
                    fake_store = fake_im.data.contiguous().view(config.batch_size*16,3,128,128)
                    torchvision.utils.save_image(atts_store, 
                        "{}att_{}.png".format(config.sample_dir,cc),normalize=True)
                    torchvision.utils.save_image(colors_store, 
                        "{}color_{}.png".format(config.sample_dir,cc),normalize=True)
                    torchvision.utils.save_image(fake_store,
                        "{}img_fake_{}.png".format(config.sample_dir,cc),normalize=True)
                    real_store = right_img.data.contiguous().view(config.batch_size * 16,3,128,128)
                    torchvision.utils.save_image(real_store,
                        "{}img_real_{}.png".format(config.sample_dir,cc),normalize=True)
                    cc += 1
                    torch.save(self.generator.state_dict(),
                               "{}/vg_net_{}.pth"
                               .format(config.model_dir,cc))
                 
                t0 = time.time()
    
    def _reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=10)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--cuda",
                        default=True)
    p
    parser.add_argument("--model_dir",
                        type=str,
                        default="./checkpoints/lele")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./result/lele/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')

    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--num_thread', type=int, default=4)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--perceptual', type=bool, default=False)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":
    config = parse_args()
    config.is_train = 'train'
    import vgnet as trainer
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)
