import os
import sys
import cv2
import argparse
import math
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import settings as settings
from dataset import ShowDataset
from model import WFDST_Net
from cal_ssim import SSIM

logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
device_ids=range(torch.cuda.device_count())
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) 


class Session:
    def __init__(self):
        self.show_dir = settings.show_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if torch.cuda.is_available():
            self.net = WFDST_Net().cuda()
        if len(device_ids) > 1:
            self.net = nn.DataParallel(WFDST_Net()).cuda()
        self.opt_net = Adam(self.net.parameters(), lr=settings.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1, settings.l2], gamma=0.1)
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = ShowDataset(dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
        return self.dataloaders[dataset_name]

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt_net.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']
        self.sche_net.last_epoch = self.step

    def inf_batch_real(self, batch):
        with torch.no_grad():
            O= batch['O'].cuda()
            O= Variable(O, requires_grad=False)
            derain = self.net(O)

        return derain, O

    def save_image(self, imgs, name):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            img_file = os.path.join(self.show_dir,'%s_derain.%s' % (name.split('.')[0],name.split('.')[-1]))
            cv2.imwrite(img_file, img)

def run_show(ckp_name_net='best_net'):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints_net(ckp_name_net)
    dataset = 'real'
    dt = sess.get_dataloader(dataset)

    for i, batch in enumerate(dt):
        logger.info(i)
        derain, O = sess.inf_batch_real( batch)
        sess.save_image(derain, batch['file_name'][0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='best_net')
    args = parser.parse_args(sys.argv[1:])
    run_show(args.model_1)

