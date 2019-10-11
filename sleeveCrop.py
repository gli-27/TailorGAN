import os
import time
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from util import util
from data import data_loader
from models import create_model
from options.options import SleeveOptions
from tensorboardX import SummaryWriter


opt = SleeveOptions().parse()
dataset = data_loader.SleeveCrop(opt)
loader = DataLoader(dataset, batch_size=512, num_workers=16, shuffle=False)
dataset_size = len(dataset)


for i, data, in enumerate(loader):
        # org_img = data
        pass
print('Finished')