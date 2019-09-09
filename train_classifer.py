import os
import time
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from util import util
from data import data_loader
from models import create_model
from options.options import Options


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("gradflow.jpg")
    plt.show()


opt = Options().parse()
print(opt)
start_epoch, epoch_iter = 1, 0

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
model.classifier.train()

total_steps = (opt.niter - start_epoch) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

total_start_time = time.time()

for epoch in range(start_epoch, opt.niter+1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(loader):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        _, _, collar_type, org_img = data
        org_img = org_img.cuda(opt.gpuid)
        collar_type = collar_type.cuda(opt.gpuid)

        # model.classifier.zero_grad()
        model.optimizer.zero_grad()
        pred_type = model.classifier(org_img)
        loss = model.loss(pred_type, collar_type)
        loss.backward()
        model.optimizer.step()
        # plot_grad_flow(model.classifier.named_parameters())

        if total_steps % opt.print_freq == print_delta:
            total_time = (time.time() - total_start_time)
            epoch_time = (time.time() - epoch_start_time)
            iter_time = (time.time() - iter_start_time)
            print('epoch: %d/%d; iters: %d/%d; total_time: %.3f; epoch_time: %.3f; iter_time: %.3f'
                  % (epoch, opt.niter, (i + 1) * opt.batch_size, dataset_size, total_time, epoch_time, iter_time))
            print('Total loss: %.5f.'
                  % loss.data)

    save_dir = opt.checkpoints_dir + '/classifier/path/'
    util.mkdir(save_dir)
    if epoch % 10 == 0:
        save_path = save_dir + 'classifier_%s_%s_%s.pth' % (opt.type_classifier, opt.resnet, epoch)
        print('Save Model.')
        torch.save(model.classifier.state_dict(), save_path)
print('Training Finished')