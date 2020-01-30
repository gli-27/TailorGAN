import os
import torch
import torch.nn as nn
from models import networks

def create_collar_model(opt):
    model = TailorGAN(opt)
    print("model [%s] was created." % (model.name()))
    return model

def create_sleeve_model(opt):
    model = SleeveGAN(opt)
    print("model [%s] was created." % (model.name()))
    return model

def create_classifier_model(opt):
    model = ClassifierModel(opt)
    print("model [%s] was created." % (model.name()))
    return model

def create_step2_model(opt):
    model = Step2(opt)
    print("model [%s] was created." % (model.name()))
    return model


class ClassifierModel(nn.Module):
    def name(self):
        return 'Classifier'

    def __init__(self, opt):
        super(ClassifierModel, self).__init__()
        self.isTrain = opt.isTrain
        if opt.type_classifier == 'collar':
            self.classifier = networks.define_classifier(opt.num_collar)
        else:
            self.classifier = networks.define_classifier(opt.num_sleeve)
        if self.isTrain:
            self.loss = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            if opt.type_classifier == 'collar':
                self.classifier.load_state_dict(torch.load('./checkpoints/classifier/path/classifier_%s_50.pth'
                                                           % opt.type_classifier,
                                                           map_location="cuda:%d" % opt.gpuid))
            else:
                self.classifier.load_state_dict(torch.load('./checkpoints/classifier/path/classifier_%s_50.pth'
                                                           % opt.type_classifier,
                                                           map_location="cuda:%d" % opt.gpuid))
            print('Model load successful!')



class TailorGAN(nn.Module):
    def name(self):
        return 'TailorGAN'

    def __init__(self, opt):
        super(TailorGAN, self).__init__()
        self.isTrain = opt.isTrain
        self.srcE = networks.define_srcEncoder(norm='instance')
        self.edgeE = networks.define_edgeEncoder(norm='instance')
        self.netG = networks.define_generator('instance', opt.n_blocks, opt.use_dropout)
        if self.isTrain:
            if opt.step == 'step2':
                self.srcE.load_state_dict(torch.load(
                    './checkpoints/Recon/collarRecon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                self.edgeE.load_state_dict(torch.load(
                    './checkpoints/Recon/collarRecon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                self.netG.load_state_dict(torch.load(
                    './checkpoints/Recon/collarRecon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                self.netD = networks.define_discriminator(opt.num_collar, input_nc=3, ndf=32, n_layers_D=3,
                                                                 norm='instance', num_D=1)
                self.optimizer_netD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*3, betas=(opt.beta1, 0.999))

                print('Model load successful!')

            # self.optimizer_srcE = torch.optim.Adam(self.srcE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_edgeE = torch.optim.Adam(self.edgeE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_netG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            params = []
            params += self.srcE.parameters()
            params += self.edgeE.parameters()
            params += self.netG.parameters()
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.class_loss = nn.CrossEntropyLoss()
            # self.recon_loss = networks.vggloss(opt)
            self.recon_loss = nn.L1Loss()
            self.concept_loss = nn.L1Loss()
            self.VGGloss = networks.vggloss(opt)
            self.adv_loss = networks.GANLOSS()
        else:
            self.edgeE.load_state_dict(torch.load(
                './checkpoints/FullModel/FullModel_collar_edgeE.pth',
                map_location="cuda:%d" % opt.gpuid
            ))
            self.srcE.load_state_dict(torch.load(
                './checkpoints/FullModel/FullModel_collar_srcE.pth',
                map_location="cuda:%d" % opt.gpuid
            ))
            self.netG.load_state_dict(torch.load(
                './checkpoints/FullModel/FullModel_collar_netG.pth',
                map_location="cuda:%d" % opt.gpuid
            ))



class SleeveGAN(nn.Module):
    def name(self):
        return 'TailorGAN'

    def __init__(self, opt):
        super(SleeveGAN, self).__init__()
        self.isTrain = opt.isTrain
        self.srcE = networks.define_srcEncoder(norm='instance')
        self.edgeE = networks.define_edgeEncoder(norm='instance')
        self.netG = networks.define_generator('instance', opt.n_blocks, opt.use_dropout)
        if self.isTrain:
            if opt.step == 'step2':
                self.srcE.load_state_dict(torch.load(
                    './checkpoints/Recon/sleeveRecon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                self.edgeE.load_state_dict(torch.load(
                    './checkpoints/Recon/sleeveRecon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                self.netG.load_state_dict(torch.load(
                    './checkpoints/Recon/sleeveRecon_srcE_%s.pth' % opt.num_epoch,
                    map_location="cuda:%d" % opt.gpuid
                ))
                self.netD = networks.define_discriminator(opt.num_sleeve, input_nc=3, ndf=32, n_layers_D=3,
                                                                 norm='instance', num_D=1)
                self.optimizer_netD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

                print('Model load successful!')
            if opt.enable_classifier:
                if opt.type_classifier == 'collar':
                    self.classifier = networks.define_classifier(opt.num_collar)
                else:
                    self.classifier = networks.define_classifier(opt.num_sleeve)
                self.classifier.load_state_dict(torch.load('./checkpoints/classifier/path/classifier_%s_%s.pth'
                                                       % (opt.type_classifier, opt.num_epoch),
                                                       map_location="cuda:%d" % opt.gpuid))
                for param in self.classifier.parameters():
                    param.requires_grad = False
            self.optimizer_srcE = torch.optim.Adam(self.srcE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_edgeE = torch.optim.Adam(self.edgeE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_netG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.class_loss = nn.CrossEntropyLoss()
            self.recon_loss = nn.L1Loss(opt)
            self.concept_loss = nn.L1Loss()
            self.VGGloss = networks.vggloss(opt)
            self.adv_loss = networks.GANLOSS()
        else:
            self.edgeE.load_state_dict(torch.load(
                './checkpoints/FullModel/FullModel_sleeve_edgeE.pth',
                map_location="cuda:%d" % opt.gpuid
            ))
            self.srcE.load_state_dict(torch.load(
                './checkpoints/FullModel/FullModel_sleeve_srcE.pth',
                map_location="cuda:%d" % opt.gpuid
            ))
            self.netG.load_state_dict(torch.load(
                './checkpoints/FullModel/FullModel_sleeve_netG.pth',
                map_location="cuda:%d" % opt.gpuid
            ))
