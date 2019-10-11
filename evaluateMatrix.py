import os
import glob
from skimage import io
from skimage.measure import compare_psnr, compare_ssim

path = './result/collarEvaluationLarge/'

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

# def computeMatrix(imgs, )

dirlist = os.listdir(path)
for fldr in dirlist:
    imgs = []
    currentPath = os.path.join(path, fldr)
    for root, dirs, files in os.walk(currentPath, topdown=True):
        for imgPath in files:
            img = io.imread(imgPath)
            imgs.append(img)

print(dirlist)