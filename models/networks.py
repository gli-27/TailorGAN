import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import torchvision.models as models

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

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

def define_C(norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    net=None
    norm_layer=get_norm_layer(norm_type=norm)
    net=classifier(norm_layer= norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G(norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    net=None
    norm_layer=get_norm_layer(norm_type=norm)
    net=generator(norm_layer= norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    net=None
    norm_layer=get_norm_layer(norm_type=norm)
    net=discriminator(norm_layer= norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


class vggLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
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


class generator(nn.Module):
    def __init__(self, n_blocks=6, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        bodyModel = [nn.ReflectionPad2d(3),
                     nn.Conv2d(3, 32, kernel_size=7, padding=0, bias=use_bias),  # 128*128
                     norm_layer(32),
                     nn.ReLU(True),
                     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 64*64
                     norm_layer(64),
                     nn.ReLU(True),
                     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 32*32
                     norm_layer(128),
                     nn.ReLU(True),
                     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 16*16
                     norm_layer(256),
                     nn.ReLU(True),
                     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 8*8
                     norm_layer(512),
                     nn.ReLU(True),
                     nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=use_bias),  ## 4*4
                     norm_layer(1024),
                     nn.ReLU(True)
                     ]
        self.bodyEncoder = nn.Sequential(*bodyModel)

        decoderModel = []
        for i in range(n_blocks):  # add ResNet blocks
            decoderModel += [
                ResnetBlock(2048, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        decoderModel += [
            # upsamp1 -> 8*8
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(True),
            # upSamp2 -> 16*16
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True),
            # upSamp3 -> 32*32
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            ##upSamp4->64*64
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
            ## tanh layer
            # nn.ReflectionPad2d(3),
            # nn.Conv2d(64, 3, kernel_size=7,padding=0),
            # nn.Tanh()
        ]
        self.decoder = nn.Sequential(*decoderModel)

        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.color = nn.Sequential(*model)
        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 1, kernel_size=7, padding=0),
                  nn.Sigmoid()]
        self.mask = nn.Sequential(*model)

    def forward(self, src_img, class_feat):

        self.bodyEncoded = self.bodyEncoder(src_img)
        vec = torch.cat((self.bodyEncoded, class_feat), dim=1)
        base = self.decoder(vec)
        color = self.color(base)
        attn = self.mask(base)
        output = attn * color + (1 - attn) * src_img
        return output  # ,attn,color


class discriminator(nn.Module):
    def __init__(self, num_classes=12, norm_layer=nn.BatchNorm2d):
        super(discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),  # 64*64
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  # 32*32
            norm_layer(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),  # 16*16
            norm_layer(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),  # 8*8
            norm_layer(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=use_bias),  # 4*4
            norm_layer(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.pool1 = nn.AvgPool2d(4, 4)
        # self.pool2 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(1024 * 4 * 4, 1024)
        # self.fc2=nn.Linear(1024*4*4,1024)
        self.adv_layer = nn.Linear(1024 * 4 * 4, 1)
        self.class_layer = nn.Linear(1024, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.classifier=nn.Linear(12800,num_classes)
        # self.avgLayer=nn.AvgPool2d(4)

    def forward(self, img):
        feature1 = self.layer1(img)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        feature5 = self.layer5(feature4)
        # feature5=feature4.view(feature4.size(0), 128*8*8)
        feature6 = feature5.view(feature4.shape[0], -1)
        x1 = self.fc1(feature6)
        # x2=F.leaky_relu(self.fc2(feature6))
        # x1 = x1.view(-1, 1024*1*1)
        # x2 =  x2.view(-1, 1024*1*1)
        # print(x2.size())
        # x2 = x2.view(x2.size(0), -1)
        # x1 = x2.view(x1.size(0), -1)
        predClass = self.class_layer(x1)
        predValid = self.adv_layer(feature6)
        predClass = self.softmax(predClass)
        predValid = self.sigmoid(predValid)
        return predValid, predClass  # , feature1, feature2, feature3, feature4


class classifier(nn.Module):
    def __init__(self, num_classes=12, norm_layer=nn.BatchNorm2d):
        super(classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 64*64
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 32*32
            norm_layer(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 16*16
            norm_layer(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 8*8
            norm_layer(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 4*4
            nn.LeakyReLU(0.2, inplace=True)
        ]
        # for i in range(4):       # add ResNet blocks
        #    model += [ResnetBlock(512, norm_layer=norm_layer,padding_type = 'reflect',  use_dropout=False,use_bias=use_bias)]
        self.layer = nn.Sequential(*model)
        # self.avgpool=nn.AvgPool2d(8)
        self.fc1 = nn.Linear(1024 * 4 * 4, 1024)

        self.fc2 = nn.Linear(1024, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, img):
        x1 = self.layer(img)
        # print(x1.shape)
        x = x1.view(-1, 1024 * 4 * 4)
        x = F.relu(self.fc1(x))
        # pred_class=self.fc2(x)
        x = F.relu(self.fc2(x))
        pred_class = self.softmax(x)
        return pred_class  # ,x1


class HED(torch.nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        self.moduleVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

    # self.load_state_dict(torch.load('./models/' + arguments_strModel + '.pytorch'))
    # end

    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tensorInput = torch.cat([tensorBlue, tensorGreen, tensorRed], 1)

        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)

        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

        tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)

        return self.moduleCombine(
            torch.cat([tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv], 1))
