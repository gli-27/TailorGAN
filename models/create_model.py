import os
import torch
import torch.nn as nn
from models import networks, networks2, network_revised

def create_model(opt):
    if opt.isTrain:
        model = TailorGAN(opt)
    else:
        pass
    print("model [%s] was created." % (model.name()))
    return model


def create_classifier_model(opt, type_classifier):
    model = ClassifierModel(opt, type_classifier)
    print("model [%s] was created." % (model.name()))
    return model


class ClassifierModel(torch.nn.Module):
    def name(self):
        return 'Classifier'

    def __init__(self, opt, type_classifier):
        super(ClassifierModel, self).__init__()
        self.isTrain = opt.isTrain
        if type_classifier == 'collar':
            self.classifier = network_revised.classifier(opt.resnet, opt.num_collar)
        else:
            self.classifier = network_revised.classifier(opt.resnet, opt.num_sleeve)
        if self.isTrain:
            self.loss = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.classifier.load_state_dict(torch.load('./checkpoints/classifier/path/classifier_%s_%s_70.pth'
                                                       % (opt.type_classifier, opt.resnet),
                                                       map_location="cuda:%d" % opt.gpuid))
            print('Model load successful!')



class TailorGAN(nn.Module):
    def name(self):
        return 'TailorGAN'

    def __init__(self, opt):
        super(TailorGAN, self).__init__()
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.srcE = network_revised.define_srcEncoder(norm='instance')
            self.edgeE = network_revised.define_edgeEncoder(norm='instance')
            self.netG = network_revised.define_generator('instance', opt.n_blocks, opt.use_dropout)
            self.netD = network_revised.define_discriminator(input_nc=3, ndf=32, n_layers_D=3, norm='instance', num_D=1)
            if opt.step == 'step2':
                self.srcE.load_state_dict(torch.load(
                    './checkpoints/TailorGAN_Garmentset/path/recon2/TailorGAN_Garment_recon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                for param in self.srcE.parameters():
                    param.requires_grad = False
                self.edgeE.load_state_dict(torch.load(
                    './checkpoints/TailorGAN_Garmentset/path/recon2/TailorGAN_Garment_recon_edgeE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                for param in self.edgeE.parameters():
                    param.requires_grad = False
                self.netG.load_state_dict(torch.load(
                    './checkpoints/TailorGAN_Garmentset/path/recon2/TailorGAN_Garment_recon_netG_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                print('Model load successful!')
            if opt.enable_classifier:
                if opt.type_classifier == 'collar':
                    self.classifier = network_revised.create_classifier(opt.resnet, opt.num_collar)
                else:
                    self.classifier = network_revised.create_classifier(opt.resnet, opt.num_sleeve)
                self.class_loss = nn.CrossEntropyLoss()
                self.classifier.load_state_dict(torch.load('./checkpoints/classifier/path/classifier_%s_%s_70.pth'
                                                       % (opt.type_classifier, opt.resnet),
                                                       map_location="cuda:%d" % opt.gpuid))
                for param in self.classifier.parameters():
                    param.requires_grad = False
            self.optimizer_srcE = torch.optim.Adam(self.srcE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_edgeE = torch.optim.Adam(self.edgeE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_netG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_netD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.recon_loss = nn.L1Loss()
            self.VGGloss = network_revised.vggloss(opt)
            self.adv_loss = network_revised.GANLoss()


"""
The train_GAN.py file model
"""
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

            self.adv_loss = networks.GANLoss()
            self.class_loss = nn.CrossEntropyLoss()


"""
The train_GAN2.py file model
"""
class TrainModel2(nn.Module):
    def name(self):
        return 'TailorGAN2'

    def __init__(self, opt):
        super(TrainModel2, self).__init__()
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.netG = networks.define_G(norm='instance')
            self.netG.load_state_dict(torch.load(os.path.join(
                opt.data_root+opt.pretrain_pkg, 'gen_vgg_inst_trans_40.ckpt'), map_location="cuda:%d" % opt.gpuid))
            self.netD = networks.define_D()
            self.classifier = networks.define_C()
            self.classifier.load_state_dict(torch.load(os.path.join(
                opt.data_root+opt.pretrain_pkg, 'class_colorImg_transOnly_79.ckpt'), map_location="cuda:%d" % opt.gpuid))
            for p in self.classifier.parameters():
                p.requires_grad = False
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=3*opt.lr)

            self.L1Loss = nn.L1Loss()
            self.adv_loss = nn.MSELoss()
            self.class_loss = nn.CrossEntropyLoss()


"""
The train_generator.py file model
"""
class TrainModel3(nn.Module):
    def name(self):
        return 'TailorGAN_generator'

    def __init__(self, opt):
        super(TrainModel3, self).__init__()

        self.netG = networks2.define_G2(norm='instance')
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
        self.L1Loss = nn.L1Loss()
        self.contentLoss = networks2.vggLoss()
        self.contentLoss.initialize(nn.MSELoss())
