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
    prev_gray, prev_y, prev_type = None, None, None
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(loader):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        gray_img, src_img, collar_type, y = data
        if i < 1:
            prev_gray, prev_y, prev_type = gray_img, y, collar_type
            continue
        valid = Tensor(y.shape[0], 1, require_grads=False).fill_(1.0).cuda()
        fake = Tensor(y.shape[0], 1, require_grads=False).fill_(0.0).cuda()
        _, class_feat = model.classifier2(prev_gray.cuda())
        fake_img = model.netG(src_img.cuda, class_feat)
        for p in model.netD.parameters():
            p.requires_grad = True
        model.optimizerD.zero_grad()
        true_valid = model.netD(y.cuda().detach())
        pred_valid = model.netD(fake_img.cuda().detach())
        d_loss = 0.5 * (model.adv_loss(true_valid, valid) + model.adv_loss(pred_valid, fake))
        d_loss.backward(retain_graph=True)
        model.optimizerD.step()

        for p in model.netD.parameters():
            p.requires_grad=False
        model.optimizerG.zero_grad()
        G_adv_loss = model.adv_loss(pred_valid, valid)
        pred_class, _ = model.classifier1(fake_img)
        G_class_loss = model.class_loss(pred_class, prev_type.cuda())
        g_loss = G_adv_loss + G_class_loss
        g_loss.backward()
        model.optimizerG.step()

        prev_gray, prev_y, prev_type = gray_img, y, collar_type

        if total_steps % opt.print_freq == print_delta:
            total_time = (time.time() - total_start_time)
            epoch_time = (time.time() - epoch_start_time)
            iter_time = (time.time() - iter_start_time)
            print('epoch: %d/%d; iters: %d/%d; total_time: %.3f; epoch_time: %.3f; iter_time: %.3f'
                  % (epoch, opt.niter, (i+1)*2, dataset_size, total_time, epoch_time, iter_time))
            print('d_loss: %.5f, G_adv_loss: %.5f, G_class_loss: %.5f'
                  % (d_loss.data, G_adv_loss.data, G_class_loss.data))

            save_fake = total_steps % opt.display_freq == display_delta

            if save_fake:
                print('save imgs')
                print('')
                path = '../result/recon/' + str(epoch) + '/' + str((i + 1) * 2)
                util.mkdir(path)
                vutils.save_image(
                    true_valid, '%s/true_valid.png' % path,
                    normalize=True
                )
                vutils.save_image(
                    pred_valid.detach(), '%s/pred_valid.png' % path,
                    normalize=True
                )

print("Train finished")