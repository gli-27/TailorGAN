import argparse
import os
from util import util

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='TailorGAN')
        self.parser.add_argument('--data_root', type=str, default='/home/cxu-serve/p1/gli27/DATA')
        self.parser.add_argument('--data_path', type=str, default='/Res/trainSet.csv')
        self.parser.add_argument('--pretrain_pkg', type=str, default='/pretrain_pkg/')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--batch_size', type=int, default=8, help='batch size of data loader.')
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--lr', type=float, default=2e-5, help='learning rate.')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--isTrain', type=bool, default=True)
        self.parser.add_argument('--gpuid', type=int, default=3)
        self.parser.add_argument('--n_blocks', type=int, default=6)
        self.parser.add_argument('--use_dropout', type=bool, default=False)
        self.parser.add_argument('--step', type=str, default='step2', help='step1 is for reconstruction, '
                                                                           'step2 is for synthesize.')
        self.parser.add_argument('--num_epoch', type=int, default=70)

        self.parser.add_argument('--type_classifier', type=str, default='collar', help='Type of classifier')
        self.parser.add_argument('--num_collar', type=int, default=12, help='Number of collar types')
        self.parser.add_argument('--num_sleeve', type=int, default=2, help='Number of collar types')
        self.parser.add_argument('--resnet', type=str, default='resnet101')
        self.parser.add_argument('--enable_classifier', type=bool, default=True, help='Use for ablation study, default'
                                                                                      'enable')

        self.parser.add_argument('--print_freq', type=int, default=10, help='batch size of training')
        self.parser.add_argument('--display_freq', type=int, default=50,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')

        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # self.opt.isTrain = self.isTrain  # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)

        return self.opt