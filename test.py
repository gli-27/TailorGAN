import time
import torch
import numpy as np
import pytorch_ssim
import torchvision.utils as vutils
from util import util
from models import create_model
from options.options import CollorOptions
from skimage.measure import compare_psnr, compare_ssim
from skimage import io
from skimage.color import rgb2gray
from torchvision import transforms

opt = CollorOptions().parse()

model = create_model.create_collar_model(opt)
model = model.cuda(opt.gpuid)
Tensor = torch.cuda.FloatTensor

class trans:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.RandomAffine(degrees=15, scale=(0.85, 1.55), translate=(0.10, 0.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.org_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

trans = trans()

for i in range(1, 9):
    path = './example/'

    edge = io.imread(path + 'edge/%02d.png' % i)
    src = io.imread(path + 'src/%02d.png' % i)

    # ref = rgb2gray(ref)

    edge = trans.transform(np.uint8(edge))
    src = trans.org_transform(src.astype(np.uint8))

    edge_tensor = edge.cuda(opt.gpuid)
    src_tensor = src.cuda(opt.gpuid)

    edge_tensor = torch.unsqueeze(edge_tensor, 0).cuda(opt.gpuid)
    src_tensor = torch.unsqueeze(src_tensor, 0).cuda(opt.gpuid)

    with torch.no_grad():
        edgeFeat = model.edgeE(edge_tensor)
        srcFeat = model.srcE(src_tensor)

        syn = model.netG(torch.cat((edgeFeat, srcFeat), dim=1), src_tensor)

        print('Save images.')
        path = './example/syn/' + str(i + 1)
        util.mkdir(path)
        vutils.save_image(
            syn, '%s/%02d.png' % (path, i),
            normalize=True
        )

