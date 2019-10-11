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
from options.options import SleeveOptions

mseloss = torch.nn.MSELoss()

def PSNR(realImg, synImg):
    maxVal = torch.max(synImg)
    mseVal = mseloss(realImg, synImg)
    return 20*np.log10(maxVal/np.sqrt(mseVal))

opt = SleeveOptions().parse()
print(opt)
start_epoch, epoch_iter = 1, 0

classifier = create_model

dataset = data_loader.SleeveTestDataset(opt)
loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
dataset_size = len(dataset)

model = create_model.create_sleeve_model(opt)
model = model.cuda(opt.gpuid)
Tensor = torch.cuda.FloatTensor

ssimshort2long = 0
ssimlong2short = 0
psnrshort2long = 0
psnrlong2short = 0


for i, data in enumerate(loader):
    for j in range(len(data)):
        data[j] = data[j].cuda(opt.gpuid)

    shortEdge, shortSrc, longEdge, longSrc, shortOrg, longOrg = data

    with torch.no_grad():
        shortEdgeFeat = model.edgeE(shortEdge)
        shortSrcFeat = model.srcE(shortSrc)
        longEdgeFeat = model.edgeE(longEdge)
        longSrcFeat = model.srcE(longSrc)

        short2long = model.netG(torch.cat((longEdgeFeat, shortSrcFeat), dim=1), shortSrc)
        long2short = model.netG(torch.cat((shortEdgeFeat, longSrcFeat), dim=1), longSrc)

        print('Save images.')
        path = './result/sleeveEvaluation/' + str(i + 1)
        util.mkdir(path)
        vutils.save_image(
            short2long, '%s/short2long.png' % path,
            normalize=True
        )
        vutils.save_image(
            long2short, '%s/long2short.png' % path,
            normalize=True
        )
        vutils.save_image(
            shortOrg, '%s/shortOrg.png' % path,
            normalize=True
        )
        vutils.save_image(
            longOrg, '%s/longOrg.png' % path,
            normalize=True
        )

        ssimshort2long += pytorch_ssim.ssim(shortOrg.cpu(), short2long.cpu())
        ssimlong2short += pytorch_ssim.ssim(longOrg.cpu(), long2short.cpu())
        psnrshort2long += PSNR(shortOrg.cpu(), short2long.cpu())
        psnrlong2short += PSNR(longOrg.cpu(), long2short.cpu())

ssimshort2long /= len(loader)
ssimlong2short /= len(loader)
psnrshort2long /= len(loader)
psnrlong2short /= len(loader)

print('ssimshort2long: %.2f' % ssimshort2long)
print('ssimlong2short: %.2f' % ssimlong2short)
print('psnrshort2long: %.2f' % psnrshort2long)
print('psnrlong2short: %.2f' % psnrlong2short)
