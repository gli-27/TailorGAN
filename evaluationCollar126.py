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
from skimage.measure import compare_psnr, compare_ssim
from skimage import io
from skimage.color import rgb2gray

mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()

def PSNR(realImg, synImg):
    maxVal = torch.max(realImg)
    mseVal = mseloss(realImg, synImg)
    return 10*np.log10(maxVal*maxVal/mseVal)

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

ssimtwo2one = 0
ssimone2two = 0
ssimone2six = 0
ssimsix2one = 0
ssimtwo2six = 0
ssimsix2two = 0

psnrtwo2one = 0
psnrone2two = 0
psnrone2six = 0
psnrsix2one = 0
psnrtwo2six = 0
psnrsix2two = 0

clstwo2one = 0
clsone2two = 0
clsone2six = 0
clssix2one = 0
clstwo2six = 0
clssix2two = 0

for i, data in enumerate(loader):
    for j in range(len(data)):
        data[j] = data[j].cuda(opt.gpuid)

    twoEdge, twoSrc, oneEdge, oneSrc, sixEdge, sixSrc,\
        twoOrg, oneOrg, sixOrg, twoType, oneType, sixType = data

    with torch.no_grad():

        twoEdgeFeat = model.edgeE(twoEdge)
        twoSrcFeat = model.srcE(twoSrc)
        oneEdgeFeat = model.edgeE(oneEdge)
        oneSrcFeat = model.srcE(oneSrc)
        sixEdgeFeat = model.edgeE(sixEdge)
        sixSrcFeat = model.srcE(sixSrc)

        two2oneSyn = model.netG(torch.cat((oneEdgeFeat, twoSrcFeat), dim=1), twoSrc)
        one2twoSyn = model.netG(torch.cat((twoEdgeFeat, oneSrcFeat), dim=1), oneSrc)
        two2sixSyn = model.netG(torch.cat((sixEdgeFeat, twoSrcFeat), dim=1), twoSrc)
        six2twoSyn = model.netG(torch.cat((twoEdgeFeat, sixSrcFeat), dim=1), sixSrc)
        one2sixSyn = model.netG(torch.cat((sixEdgeFeat, oneSrcFeat), dim=1), oneSrc)
        six2oneSyn = model.netG(torch.cat((oneEdgeFeat, sixSrcFeat), dim=1), sixSrc)

        """
        two2oneSyn = model.netG(oneEdge, twoSrc)
        one2twoSyn = model.netG(twoEdge, oneSrc)
        two2sixSyn = model.netG(sixEdge, twoSrc)
        six2twoSyn = model.netG(twoEdge, sixSrc)
        one2sixSyn = model.netG(sixEdge, oneSrc)
        six2oneSyn = model.netG(oneEdge, sixSrc)
        """

        # cls = classifier(two2oneSyn)
        clstwo2one += celoss(classifier.classifier(two2oneSyn), oneType)
        clsone2two += celoss(classifier.classifier(one2twoSyn), twoType)
        clstwo2six += celoss(classifier.classifier(two2sixSyn), sixType)
        clssix2two += celoss(classifier.classifier(six2twoSyn), twoType)
        clsone2six += celoss(classifier.classifier(one2sixSyn), sixType)
        clssix2one += celoss(classifier.classifier(six2oneSyn), oneType)

        """
        ssimtwo2one += pytorch_ssim.ssim(two2oneSyn.cpu(), twoOrg.cpu())
        ssimone2two += pytorch_ssim.ssim(one2twoSyn.cpu(), oneOrg.cpu())
        ssimone2six += pytorch_ssim.ssim(one2sixSyn.cpu(), oneOrg.cpu())
        ssimsix2one += pytorch_ssim.ssim(six2oneSyn.cpu(), sixOrg.cpu())
        ssimtwo2six += pytorch_ssim.ssim(two2sixSyn.cpu(), twoOrg.cpu())
        ssimsix2two += pytorch_ssim.ssim(six2twoSyn.cpu(), sixOrg.cpu())

        psnrtwo2one += PSNR(twoOrg.cpu(), two2oneSyn.cpu())
        psnrone2two += PSNR(oneOrg.cpu(), one2twoSyn.cpu())
        psnrone2six += PSNR(oneOrg.cpu(), one2sixSyn.cpu())
        psnrsix2one += PSNR(sixOrg.cpu(), six2oneSyn.cpu())
        psnrtwo2six += PSNR(twoOrg.cpu(), two2sixSyn.cpu())
        psnrsix2two += PSNR(sixOrg.cpu(), six2twoSyn.cpu())
        """
        """
        ssimtwo2one += compare_ssim(two2oneSyn.cpu().data.numpy(), twoOrg.cpu().data.numpy(), multichannel=True)
        ssimone2two += compare_ssim(one2twoSyn.cpu().data.numpy(), oneOrg.cpu().data.numpy(), multichannel=True)
        ssimone2six += compare_ssim(one2sixSyn.cpu().data.numpy(), oneOrg.cpu().data.numpy(), multichannel=True)
        ssimsix2one += compare_ssim(six2oneSyn.cpu().data.numpy(), sixOrg.cpu().data.numpy(), multichannel=True)
        ssimtwo2six += compare_ssim(two2sixSyn.cpu().data.numpy(), twoOrg.cpu().data.numpy(), multichannel=True)
        ssimsix2two += compare_ssim(six2twoSyn.cpu().data.numpy(), sixOrg.cpu().data.numpy(), multichannel=True)

        psnrtwo2one += compare_psnr(twoOrg.cpu().data.numpy(), two2oneSyn.cpu().data.numpy())
        psnrone2two += compare_psnr(oneOrg.cpu().data.numpy(), one2twoSyn.cpu().data.numpy())
        psnrone2six += compare_psnr(oneOrg.cpu().data.numpy(), one2sixSyn.cpu().data.numpy())
        psnrsix2one += compare_psnr(sixOrg.cpu().data.numpy(), six2oneSyn.cpu().data.numpy())
        psnrtwo2six += compare_psnr(twoOrg.cpu().data.numpy(), two2sixSyn.cpu().data.numpy())
        psnrsix2two += compare_psnr(sixOrg.cpu().data.numpy(), six2twoSyn.cpu().data.numpy())
        """

        print('Save images.')
        path = './result/collarEvaluationLarge/' + str(i + 1)
        util.mkdir(path)
        vutils.save_image(
            two2oneSyn, '%s/two2oneSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            one2twoSyn, '%s/one2twoSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            one2sixSyn, '%s/one2sixSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            six2oneSyn, '%s/six2oneSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            two2sixSyn, '%s/two2sixSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            six2twoSyn, '%s/six2twoSyn.png' % path,
            normalize=True
        )
        vutils.save_image(
            twoOrg, '%s/twoOrg.png' % path,
            normalize=True
        )
        vutils.save_image(
            oneOrg, '%s/oneOrg.png' % path,
            normalize=True
        )
        vutils.save_image(
            sixOrg, '%s/sixOrg.png' % path,
            normalize=True
        )

        two2oneSyn = io.imread('%s/two2oneSyn.png' % path)
        one2twoSyn = io.imread('%s/one2twoSyn.png' % path)
        one2sixSyn = io.imread('%s/one2sixSyn.png' % path)
        six2oneSyn = io.imread('%s/six2oneSyn.png' % path)
        two2sixSyn = io.imread('%s/two2sixSyn.png' % path)
        six2twoSyn = io.imread('%s/six2twoSyn.png' % path)
        twoOrg = io.imread('%s/twoOrg.png' % path)
        oneOrg = io.imread('%s/oneOrg.png' % path)
        sixOrg = io.imread('%s/sixOrg.png' % path)

        ssimtwo2one += compare_ssim(two2oneSyn, twoOrg, multichannel=True)
        ssimone2two += compare_ssim(one2twoSyn, oneOrg, multichannel=True)
        ssimone2six += compare_ssim(one2sixSyn, oneOrg, multichannel=True)
        ssimsix2one += compare_ssim(six2oneSyn, sixOrg, multichannel=True)
        ssimtwo2six += compare_ssim(two2sixSyn, twoOrg, multichannel=True)
        ssimsix2two += compare_ssim(six2twoSyn, sixOrg, multichannel=True)

        twoOrgGray = rgb2gray(twoOrg)
        oneOrgGray = rgb2gray(oneOrg)
        sixOrgGray = rgb2gray(sixOrg)
        two2oneGray = rgb2gray(two2oneSyn)
        one2twoGray = rgb2gray(one2twoSyn)
        one2sixGray = rgb2gray(one2sixSyn)
        six2oneGray = rgb2gray(six2oneSyn)
        two2sixGray = rgb2gray(two2sixSyn)
        six2twoGray = rgb2gray(six2twoSyn)

        psnrtwo2one += compare_psnr(twoOrgGray, two2oneGray)
        psnrone2two += compare_psnr(oneOrgGray, one2twoGray)
        psnrone2six += compare_psnr(oneOrgGray, one2sixGray)
        psnrsix2one += compare_psnr(sixOrgGray, six2oneGray)
        psnrtwo2six += compare_psnr(twoOrgGray, two2sixGray)
        psnrsix2two += compare_psnr(sixOrgGray, six2twoGray)

ssimtwo2one /= len(loader)
ssimone2two /= len(loader)
ssimone2six /= len(loader)
ssimsix2one /= len(loader)
ssimtwo2six /= len(loader)
ssimsix2two /= len(loader)
print('ssimtwo2one: %.2f' % ssimtwo2one)
print('ssimone2two: %.2f' % ssimone2two)
print('ssimone2six: %.2f' % ssimone2six)
print('ssimsix2one: %.2f' % ssimsix2one)
print('ssimtwo2six: %.2f' % ssimtwo2six)
print('ssimsix2two: %.2f' % ssimsix2two)

psnrtwo2one /= len(loader)
psnrone2two /= len(loader)
psnrone2six /= len(loader)
psnrsix2one /= len(loader)
psnrtwo2six /= len(loader)
psnrsix2two /= len(loader)
print('psnrtwo2one: %.2f' % psnrtwo2one)
print('psnrone2two: %.2f' % psnrone2two)
print('psnrone2six: %.2f' % psnrone2six)
print('psnrsix2one: %.2f' % psnrsix2one)
print('psnrtwo2six: %.2f' % psnrtwo2six)
print('psnrsix2two: %.2f' % psnrsix2two)

clstwo2one /= len(loader)
clsone2two /= len(loader)
clsone2six /= len(loader)
clssix2one /= len(loader)
clstwo2six /= len(loader)
clssix2two /= len(loader)
print('clstwo2one: %.2f' % clstwo2one)
print('clsone2two: %.2f' % clsone2two)
print('clsone2six: %.2f' % clsone2six)
print('clssix2one: %.2f' % clssix2one)
print('clstwo2six: %.2f' % clstwo2six)
print('clssix2two: %.2f' % clssix2two)
