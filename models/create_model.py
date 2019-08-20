import os
import torch
import torch.nn as nn
from models import networks

def create_model(opt):
    if opt.isTrain:
        model = TrainModel(opt)
    else:
        pass
    print("model [%s] was created." % (model.name()))
    return model

class TrainModel(torch.nn.Module):
    def name(self):
        return 'TailorGAN'

    def __init__(self, opt):
        super(TrainModel, self).__init__()
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.netG = networks.define_G(norm='instance')
            self.netG.load_state_dict(torch.load(os.path.join(
                opt.data_root+opt.pretrain_pkg, 'gen_vgg_inst_trans_40.ckpt'), map_location="cuda:%d" % opt.gpuid))
            self.netD = networks.define_D()
            self.classifier1 = networks.define_C()
            self.classifier1.load_state_dict(torch.load(os.path.join(
                opt.data_root+opt.pretrain_pkg, 'class_colorImg_transOnly_79.ckpt'), map_location="cuda:%d" % opt.gpuid))
            for p in self.classifier1.parameters():
                p.requires_grad = False
            self.classifier2 = networks.define_C()
            self.classifier2.load_state_dict(torch.load(os.path.join(
                opt.data_root+opt.pretrain_pkg, 'classifier_trans_79.ckpt'), map_location="cuda:%d" % opt.gpuid))
            for p in self.classifier2.parameters():
                p.requires_grad = False

            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=3*opt.lr)

            self.adv_loss = nn.MSELoss()
            self.class_loss = nn.CrossEntropyLoss()
