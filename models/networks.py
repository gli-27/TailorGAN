import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torchvision.models as models
from torch.nn import init
from torch.autograd import Variable

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_srcEncoder(norm='batch'):
    norm_layer = get_norm_layer(norm_type=norm)
    net = srcEncoder(norm_layer=norm_layer)
    print(net)
    net.apply(weights_init)
    return net

def define_edgeEncoder(norm='batch'):
    norm_layer = get_norm_layer(norm_type=norm)
    net = edgeEncoder(norm_layer=norm_layer)
    print(net)
    net.apply(weights_init)
    return net

def define_generator(norm='batch', n_blocks=6, use_dopout=False, padding_type='reflect'):
    norm_layer = get_norm_layer(norm_type=norm)
    net = generator(n_blocks, norm_layer, use_dopout, padding_type)
    print(net)
    net.apply(weights_init)
    return net

def define_discriminator(num_class, input_nc, ndf, n_layers_D, norm='batch', use_sigmoid=False, num_D=3):
    norm_layer = get_norm_layer(norm_type=norm)
    # netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D)
    netD = TailorDiscriminator(num_class, input_nc, norm_layer)
    print(netD)
    netD.apply(weights_init)
    return netD

def define_classifier(num_classes):
    net = Classifier(num_classes)
    print(net)
    return net

class GANLOSS(nn.Module):
    def __init__(self):
        super(GANLOSS, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def get_tensor(self, size, target_is_real):
        if target_is_real:
            return Variable(torch.ones(size, 1), requires_grad=False)
        else:
            return Variable(torch.zeros(size, 1), requires_grad=False)

    def __call__(self, input, target_is_real, gpuid):
        target_tensor = self.get_tensor(input.size(0), target_is_real).cuda(gpuid)
        return self.criterion(input, target_tensor)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, gpuid):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real).cuda(gpuid)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real).cuda(gpuid)
            return self.loss(input[-1], target_tensor)

class vggloss(nn.Module):
    def __init__(self, opt):
        super(vggloss, self).__init__()
        self.vgg = Vgg19().cuda(opt.gpuid)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class edgeEncoder(nn.Module):
    # This is the partEncoder of original form, and this will encode the edge map.
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(edgeEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        edge_encoder = [nn.ReflectionPad2d(3),
                        nn.Conv2d(3, 32, kernel_size=7, padding=0, bias=use_bias),  # 128*128
                        norm_layer(32),
                        nn.ReLU(True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 64*64
                        norm_layer(64),
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 32*32
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 16*16
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 8*8
                        norm_layer(256),
                        nn.ReLU(True),
                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 4*4
                        norm_layer(512),
                        nn.ReLU(True)
                        ]
        self.edge_encoder = nn.Sequential(*edge_encoder)

    def forward(self, edge_img):
        edge_feat = self.edge_encoder(edge_img)
        return edge_feat


class srcEncoder(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(srcEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        src_encoder = [nn.ReflectionPad2d(3),
                        nn.Conv2d(3, 32, kernel_size=7, padding=0, bias=use_bias),  # 128*128
                        norm_layer(32),
                        nn.ReLU(True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 64*64
                        norm_layer(64),
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 32*32
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 16*16
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 8*8
                        norm_layer(256),
                        nn.ReLU(True),
                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 4*4
                        norm_layer(512),
                        nn.ReLU(True)
                        ]
        self.src_encoder = nn.Sequential(*src_encoder)

    def forward(self, src_img):
        return self.src_encoder(src_img)


class generator(nn.Module):
    def __init__(self, n_blocks=6, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        decoderModel = []
        for i in range(n_blocks):  # add ResNet blocks
            decoderModel += [
                ResnetBlock(1024, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        decoderModel += [
            # upsamp1 -> 8*8
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True),
            # upSamp2 -> 16*16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            # upSamp3 -> 32*32
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            ##upSamp4->64*64
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        ]
        self.decoder = nn.Sequential(*decoderModel)

        color = []
        color += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.color = nn.Sequential(*color)

        mask = []
        mask += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 1, kernel_size=7, padding=0),
                  nn.Sigmoid()]
        self.mask = nn.Sequential(*mask)

    def forward(self, feat, src_img=None):
        row_img = self.decoder(feat)
        if src_img is not None:
            color = self.color(row_img)
            attn = self.mask(row_img)
            output_img = attn * color + (1 - attn) * src_img
            return output_img
        else:
            return row_img

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        self.netD = []
        netD = [
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        ]
        self.netD = nn.Sequential(*netD)
        self.fc1 = nn.Linear(512*4*4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.netD(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)

class TailorDiscriminator(nn.Module):
    def __init__(self, num_class, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(TailorDiscriminator, self).__init__()
        self.netD = []
        netD = [
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        ]
        self.netD = nn.Sequential(*netD)
        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.classifier = nn.Sequential(*[
            nn.Linear(256, num_class),
            # nn.Softmax(dim=1)
        ])
        self.dis = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.netD(x)
        feat1 = feat.view(feat.size(0), -1)
        feat1 = self.fc1(feat1)
        pred_class = self.classifier(feat1)
        dis = self.dis(feat1)
        return pred_class, dis, feat

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        return self.model(input)

class Classifier(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()
        self.netD = []
        netD = [
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        ]
        self.netD = nn.Sequential(*netD)
        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.classifier = nn.Sequential(*[
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=1)
        ])

    def forward(self, x):
        feat = self.netD(x)
        feat1 = feat.view(feat.size(0), -1)
        feat1 = self.fc1(feat1)
        pred_class = self.classifier(feat1)
        return pred_class


from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
