import os
import time
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from util import util
from data import data_loader
from models import create_model
from options.options import Options


opt = Options().parse()
print(opt)

dataset = data_loader.FashionDataset(opt)
loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
dataset_size = len(dataset)

type_classifier = 'collar'
if type_classifier == 'collar':
    type_list = [0] * 12
    type_list = torch.Tensor(type_list).cuda(opt.gpuid)
else:
    type_list = [0] * 2
    type_list = torch.Tensor(type_list).cuda(opt.gpuid)

model = create_model.create_classifier_model(opt, type_classifier)
model = model.cuda(opt.gpuid)

total = 0.0
correct = 0.0
with torch.no_grad():
    for i, data in enumerate(loader):
        _, _, collar_type, org_img = data
        org_img = org_img.cuda(opt.gpuid)
        collar_type = collar_type.cuda(opt.gpuid)

        pred_type = model.classifier(org_img)
        _, predicted = torch.max(pred_type.data, 1)
        total += collar_type.size(0)
        correct += (predicted == collar_type).sum().item()
        print('accuracy: %.3f %%' % (100 * correct/total))

print('Accuracy of the network on the dataset is: %d %%' % (100 * correct/total))
