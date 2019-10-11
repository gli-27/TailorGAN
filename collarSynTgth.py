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
from options.options import CollorOptions
# from tensorboardX import SummaryWriter

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

# writer = SummaryWriter('./log_files/')

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

        edge_img, refer_edge_img, src_img, org_img_type, refer_img_type, org_img, refer_org_img = data

        edge_img = edge_img.cuda(opt.gpuid)
        refer_edge_img = refer_edge_img.cuda(opt.gpuid)
        src_img = src_img.cuda(opt.gpuid)
        org_img_type = org_img_type.cuda(opt.gpuid)
        refer_img_type = refer_img_type.cuda(opt.gpuid)
        org_img = org_img.cuda(opt.gpuid)
        refer_org_img = refer_org_img.cuda(opt.gpuid)

        # Train discriminator
        for param in model.netD.parameters():
            param.requires_grad = True
        model.optimizer_netD.zero_grad()
        syn_img = model.netG(refer_edge_img, src_img)
        true_pred_class, org_img_d, a = model.netD(org_img)
        fake_pred_class, syn_img_d, b = model.netD(syn_img)
        true_pred_loss = model.class_loss(true_pred_class, org_img_type)
        # fake_pred_loss = model.class_loss(fake_pred_class, refer_img_type)
        lossD_real = model.adv_loss(org_img_d, True, opt.gpuid)
        lossD_fake = model.adv_loss(syn_img_d, False, opt.gpuid)
        dis_class_loss = true_pred_loss
        lossD = 0.5 * lossD_real + 0.5 * lossD_fake + dis_class_loss
        lossD.backward()
        model.optimizer_netD.step()

        # Synthesize step
        # model.optimizer_edgeE.zero_grad()
        # model.optimizer_srcE.zero_grad()
        for param in model.netD.parameters():
            param.requires_grad = False
        # model.optimizer_netD.zero_grad()
        model.optimizer_netG.zero_grad()
        syn_img = model.netG(refer_edge_img, src_img)
        syn_class_d, syn_img_d, syn_feat_d = model.netD(syn_img)
        refer_class_d, _, refer_feat_d = model.netD(refer_org_img)
        ganloss = model.adv_loss(syn_img_d, True, opt.gpuid)
        # syn_class, _ = model.classifier(syn_img)
        # _, real_feat = model.classifier(refer_org_img)
        classifierloss = model.class_loss(syn_class_d, refer_img_type)
        conceptloss = model.concept_loss(syn_feat_d, refer_feat_d)
        VGGloss = model.VGGloss(syn_img, org_img)
        loss = ganloss * 2 + VGGloss * 5 + conceptloss * 2 + classifierloss * 0.2
        loss.backward()
        model.optimizer_netG.step()

        # model.optimizer_srcE.step()
        # model.optimizer_edgeE.step()

        if total_steps % opt.print_freq == print_delta:
            total_time = (time.time() - total_start_time)
            epoch_time = (time.time() - epoch_start_time)
            iter_time = (time.time() - iter_start_time)
            print('epoch: %d/%d; iters: %d/%d; total_time: %.3f; epoch_time: %.3f; iter_time: %.3f'
                  % (epoch, opt.niter, (i+1)*opt.batch_size, dataset_size, total_time, epoch_time, iter_time))
            print('Total loss: %.5f; discriminatorloss: %.5f; ganloss: %.5f; VGGloss: %.5f; '
                  'conceptloss: %.5f; classifierloss: %.5f.'
                  % (loss.data, lossD.data, ganloss.data * 2, VGGloss.data * 5, conceptloss.data * 2, classifierloss.data * 0.2))
            # print('Total loss: %.5f; ganloss: %.5f; '
            #       'discriminatorloss: %.5f'
            #       % (loss.data, ganloss.data, lossD.data))

            # writer.add_scalar('Val/GANLoss', ganloss, epoch)
            # writer.add_scalar('Val/Classloss', classloss, epoch)
            # writer.add_scalar('Val/VGGloss', VGGloss, epoch)

        save_fake = total_steps % opt.display_freq == display_delta

        if save_fake:
            print('save imgs')
            print('')
            path = './result/collarGSyn/' + str(epoch) + '/' + str((i + 1) * opt.batch_size)
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
                refer_org_img, '%s/refer_imgs.png' % path,
                normalize=True
            )
            vutils.save_image(
                syn_img.detach(), '%s/syn_imgs.png' % path,
                normalize=True
            )

    save_dir = opt.checkpoints_dir + '/TailorGAN_Garmentset/path/collarG/Syn/'
    util.mkdir(save_dir)

    if epoch % 6 == 0:
        save_path_netG = save_dir + 'TailorGAN_Garment_syn_netG_%s.pth' % epoch
        torch.save(model.netG.state_dict(), save_path_netG)
        save_path_netD = save_dir + 'TailorGAN_Garment_syn_netD_%s.pth' % epoch
        torch.save(model.netD.state_dict(), save_path_netD)
        print('Model saved!')

print('Training Finished')