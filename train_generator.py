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
start_epoch, epoch_iter = 1, 0

dataset = data_loader.FashionDataset(opt)
loader = DataLoader(dataset, batch_size=opt.batch_size)
dataset_size = len(dataset)

model = create_model.create_model(opt)
model = model.cuda()
Tensor = torch.cuda.FloatTensor

total_steps = (opt.niter - start_epoch) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

total_start_time = time.time()

for epoch in range(start_epoch, opt.niter):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(loader):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        gray_img, src_img, _, y = data

        gray_img = gray_img.cuda()
        src_img = src_img.cuda()
        y = y.cuda()

        outputs = model.netG(src_img, gray_img)
        loss = model.contentLoss.get_loss(outputs, y.float())
        model.optimizerG.zero_grad()
        loss.backward()
        model.optimizerG.step()

        if total_steps % opt.print_freq == print_delta:
            total_time = (time.time() - total_start_time)
            epoch_time = (time.time() - epoch_start_time)
            iter_time = (time.time() - iter_start_time)
            print('epoch: %d/%d; iters: %d/%d; total_time: %.3f; epoch_time: %.3f; iter_time: %.3f'
                  % (epoch, opt.niter, (i+1)*2, dataset_size, total_time, epoch_time, iter_time))
            print('loss: %.5f'
                  % loss.data)

            save_fake = total_steps % opt.display_freq == display_delta

            if save_fake:
                print('save imgs')
                print('')
                path = './result/Generator/' + str(epoch) + '/' + str((i + 1) * 2)
                util.mkdir(path)
                vutils.save_image(
                    src_img, '%s/src_img.png' % path,
                    normalize=True
                )
                vutils.save_image(
                    outputs.detach(), '%s/fake_img.png' % path,
                    normalize=True
                )

print("Train finished")