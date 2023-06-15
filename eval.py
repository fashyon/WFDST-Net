import os
import sys
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import settings as settings
from dataset import TestDataset
from model import WFDST_Net
from cal_ssim import SSIM


logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
device_ids=range(torch.cuda.device_count())
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)

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
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if torch.cuda.is_available():
            self.net = WFDST_Net().cuda()
        if len(device_ids) > 1:
            self.net = nn.DataParallel(WFDST_Net()).cuda()
        self.opt_net = Adam(self.net.parameters(), lr=settings.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1, settings.l2], gamma=0.1)
        self.l2 = MSELoss().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1, drop_last=False)
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

    def inf_batch(self, name, batch):
        with torch.no_grad():
            O, B = batch['O'].cuda(), batch['B'].cuda()
            R = O - B
            O, B, R = Variable(O, requires_grad=False), Variable(B, requires_grad=False), Variable(R, requires_grad=False)
            O_derain = self.net(O)
            torch.cuda.synchronize()

        loss_list = [self.l2(O_derain, B)-self.ssim(O_derain, B) for O_derain in [O_derain]]
        ssim_list = [self.ssim(O_derain, B) for O_derain in [O_derain]]
        psnr = PSNR(O_derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)

        return losses, psnr


def run_test(ckp_name_net='latest_net'):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints_net(ckp_name_net)
    dt = sess.get_dataloader('test')
    psnr_all = 0
    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        losses, psnr= sess.inf_batch('test', batch)
        psnr_all = psnr_all + psnr
        batch_size = batch['O'].size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d %s: %f' % (i, key, val))
        logger.info('batch %d psnr: %f' % (i, psnr))
    for key, val in all_losses.items():
        logger.info('total average %s: %f' % (key, val / all_num))
    print('total average psnr:%8f' % (psnr_all / all_num))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='best_net')
    args = parser.parse_args(sys.argv[1:])
    run_test(args.model_1)

