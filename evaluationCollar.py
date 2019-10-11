import time
import torch
import numpy as np
import pytorch_ssim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from util import util
from data import data_loader
from models import create_model
from options.options import CollorOptions

mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()

def PSNR(realImg, synImg):
    maxVal = 1
    mseVal = mseloss(realImg, synImg)
    return 20*np.log10(maxVal/np.sqrt(mseVal))

opt = CollorOptions().parse()
print(opt)
start_epoch, epoch_iter = 1, 0

dataset = data_loader.CollarTestDataset(opt)
loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
dataset_size = len(dataset)

model = create_model.create_collar_model(opt)
model = model.cuda(opt.gpuid)
Tensor = torch.cuda.FloatTensor

classifier = create_model.create_classifier_model(opt)
classifier = classifier.cuda(opt.gpuid)

ssimzero2one = 0
ssimone2zero = 0
ssimone2five = 0
ssimfive2one = 0
ssimzero2five = 0
ssimfive2zero = 0

psnrzero2one = 0
psnrone2zero = 0
psnrone2five = 0
psnrfive2one = 0
psnrzero2five = 0
psnrfive2zero = 0

clszero2one = 0
clsone2zero = 0
clsone2five = 0
clsfive2one = 0
clszero2five = 0
clsfive2zero = 0

for i, data in enumerate(loader):
    for j in range(len(data)):
        data[j] = data[j].cuda(opt.gpuid)

    zeroEdge, zeroSrc, oneEdge, oneSrc, fiveEdge, fiveSrc,\
        zeroOrg, oneOrg, fiveOrg, zeroType, oneType, fiveType = data

    with torch.no_grad():
        zeroEdgeFeat = model.edgeE(zeroEdge)
        zeroSrcFeat = model.srcE(zeroSrc)
        oneEdgeFeat = model.edgeE(oneEdge)
        oneSrcFeat = model.srcE(oneSrc)
        fiveEdgeFeat = model.edgeE(fiveEdge)
        fiveSrcFeat = model.srcE(fiveSrc)

        zero2oneSyn = model.netG(torch.cat((oneEdgeFeat, zeroSrcFeat), dim=1), zeroSrc)
        one2zeroSyn = model.netG(torch.cat((zeroEdgeFeat, oneSrcFeat), dim=1), oneSrc)
        zero2fiveSyn = model.netG(torch.cat((fiveEdgeFeat, zeroSrcFeat), dim=1), zeroSrc)
        five2zeroSyn = model.netG(torch.cat((zeroEdgeFeat, fiveSrcFeat), dim=1), fiveSrc)
        one2fiveSyn = model.netG(torch.cat((fiveEdgeFeat, oneSrcFeat), dim=1), oneSrc)
        five2oneSyn = model.netG(torch.cat((oneEdgeFeat, fiveSrcFeat), dim=1), fiveSrc)

        zero2oneSyn = model.netG(oneEdge, zeroSrc)
        one2zeroSyn = model.netG(zeroEdge, oneSrc)
        zero2fiveSyn = model.netG(fiveEdge, zeroSrc)
        five2zeroSyn = model.netG(zeroEdge, fiveSrc)
        one2fiveSyn = model.netG(fiveEdge, oneSrc)
        five2oneSyn = model.netG(oneEdge, fiveSrc)

        # cls = classifier(zero2oneSyn)
        clszero2one += celoss(classifier.classifier(zero2oneSyn), oneType)
        clsone2zero += celoss(classifier.classifier(one2zeroSyn), zeroType)
        clszero2five += celoss(classifier.classifier(zero2fiveSyn), fiveType)
        clsfive2zero += celoss(classifier.classifier(five2zeroSyn), zeroType)
        clsone2five += celoss(classifier.classifier(one2fiveSyn), fiveType)
        clsfive2one += celoss(classifier.classifier(five2oneSyn), oneType)

        ssimzero2one += pytorch_ssim.ssim(zero2oneSyn.cpu(), zeroOrg.cpu())
        ssimone2zero += pytorch_ssim.ssim(one2zeroSyn.cpu(), oneOrg.cpu())
        ssimone2five += pytorch_ssim.ssim(one2fiveSyn.cpu(), oneOrg.cpu())
        ssimfive2one += pytorch_ssim.ssim(five2oneSyn.cpu(), fiveOrg.cpu())
        ssimzero2five += pytorch_ssim.ssim(zero2fiveSyn.cpu(), zeroOrg.cpu())
        ssimfive2zero += pytorch_ssim.ssim(five2zeroSyn.cpu(), fiveOrg.cpu())

        psnrzero2one += PSNR(zeroOrg.cpu(), zero2oneSyn.cpu())
        psnrone2zero += PSNR(oneOrg.cpu(), one2zeroSyn.cpu())
        psnrone2five += PSNR(oneOrg.cpu(), one2fiveSyn.cpu())
        psnrfive2one += PSNR(fiveOrg.cpu(), five2oneSyn.cpu())
        psnrzero2five += PSNR(zeroOrg.cpu(), zero2fiveSyn.cpu())
        psnrfive2zero += PSNR(fiveOrg.cpu(), five2zeroSyn.cpu())

        print('Save images.')
        path = './result/collarEvaluationLarge/' + str(i + 1)
        util.mkdir(path)
        vutils.save_image(
            zero2oneSyn, '%s/zero2oneSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            one2zeroSyn, '%s/one2zeroSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            one2fiveSyn, '%s/one2fiveSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            five2oneSyn, '%s/five2oneSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            zero2fiveSyn, '%s/zero2fiveSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            five2zeroSyn, '%s/five2zeroSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            zeroOrg, '%s/zeroOrg.png' % path,
            normalize=True
        )
        vutils.save_image(
            oneOrg, '%s/oneOrg.png' % path,
            normalize=True
        )
        vutils.save_image(
            fiveOrg, '%s/fiveOrg.png' % path,
            normalize=True
        )

ssimzero2one /= len(loader)
ssimone2zero /= len(loader)
ssimone2five /= len(loader)
ssimfive2one /= len(loader)
ssimzero2five /= len(loader)
ssimfive2zero /= len(loader)
print('ssimzero2one: %.2f' % ssimzero2one)
print('ssimone2zero: %.2f' % ssimone2zero)
print('ssimone2five: %.2f' % ssimone2five)
print('ssimfive2one: %.2f' % ssimfive2one)
print('ssimzero2five: %.2f' % ssimzero2five)
print('ssimfive2zero: %.2f' % ssimfive2zero)

psnrzero2one /= len(loader)
psnrone2zero /= len(loader)
psnrone2five /= len(loader)
psnrfive2one /= len(loader)
psnrzero2five /= len(loader)
psnrfive2zero /= len(loader)
print('psnrzero2one: %.2f' % psnrzero2one)
print('psnrone2zero: %.2f' % psnrone2zero)
print('psnrone2five: %.2f' % psnrone2five)
print('psnrfive2one: %.2f' % psnrfive2one)
print('psnrzero2five: %.2f' % psnrzero2five)
print('psnrfive2zero: %.2f' % psnrfive2zero)

clszero2one /= len(loader)
clsone2zero /= len(loader)
clsone2five /= len(loader)
clsfive2one /= len(loader)
clszero2five /= len(loader)
clsfive2zero /= len(loader)
print('clszero2one: %.2f' % clszero2one)
print('clsone2zero: %.2f' % clsone2zero)
print('clsone2five: %.2f' % clsone2five)
print('clsfive2one: %.2f' % clsfive2one)
print('clszero2five: %.2f' % clszero2five)
print('clsfive2zero: %.2f' % clsfive2zero)
