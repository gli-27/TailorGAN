import os
import time
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from util import util
from data import data_loader
from models import create_model
from options.options import CollorOptions

opt = CollorOptions().parse()
print(opt)
start_epoch, epoch_iter = 1, 0

dataset = data_loader.InterDataset(opt)
loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
dataset_size = len(dataset)

model = create_model.create_collar_model(opt)
model = model.cuda(opt.gpuid)
Tensor = torch.cuda.FloatTensor

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

        edge_img, _, src_img, img_type, _, org_img, _ = data

        edge_img = edge_img.cuda(opt.gpuid)
        src_img = src_img.cuda(opt.gpuid)
        img_type = img_type.cuda(opt.gpuid)
        org_img = org_img.cuda(opt.gpuid)

        # Reconstruction step
        model.optimizer_netG.zero_grad()
        recon_img = model.netG(edge_img, src_img)
        recon_loss = model.recon_loss(recon_img, org_img)
        recon_loss.backward()
        model.optimizer_netG.step()

        if total_steps % opt.print_freq == print_delta:
            total_time = (time.time() - total_start_time)
            epoch_time = (time.time() - epoch_start_time)
            iter_time = (time.time() - iter_start_time)
            print('epoch: %d/%d; iters: %d/%d; total_time: %.3f; epoch_time: %.3f; iter_time: %.3f'
                  % (epoch, opt.niter, (i+1)*opt.batch_size, dataset_size, total_time, epoch_time, iter_time))
            print('reconl1loss: %.5f'
                  % recon_loss.data)

        save_fake = total_steps % opt.display_freq == display_delta

        if save_fake:
            print('save imgs')
            print('')
            path = './result/collarReconTogether/' + str(epoch) + '/' + str((i + 1) * opt.batch_size)
            util.mkdir(path)
            vutils.save_image(
                org_img, '%s/org_imgs.png' % path,
                normalize=True
            )
            vutils.save_image(
                src_img, '%s/src_imgs.png' % path,
                normalize=True
            )
            vutils.save_image(
                edge_img, '%s/edge_imgs.png' % path,
                normalize=True
            )
            vutils.save_image(
                recon_img.detach(), '%s/recon_imgs.png' % path,
                normalize=True
            )

    save_dir = opt.checkpoints_dir + '/TailorGAN_Garmentset/path/collarG/Recon/'
    util.mkdir(save_dir)
    if epoch % 20 == 0:
        save_path_srcE = save_dir + 'TailorGAN_Garment_collar_recon_%s.pth' % epoch
        torch.save(model.netG.state_dict(), save_path_srcE)
        print('Model saved!')

print('Training Finished')