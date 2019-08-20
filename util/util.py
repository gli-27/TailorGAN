import os
import numpy as np
import torch
import torchvision.transforms.functional as F

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gaussianNoisy(mu, std, size):
    return np.random.normal(mu, std, size)

def rgb2gray(img):
    img = img.cpu()
    if len(img.shape) == 4:
        b, _, h, w = img.size()
        gray = np.zeros([b, 1, h, w])
        for i in range(0, b):
            temp = F.to_pil_image(img[i, :, :, :])
            temp = F.to_grayscale(temp)
            gray[i,:,:,:] = temp
        return gray
    else:
        temp = F.to_pil_image(img)
        return F.to_grayscale(temp)

