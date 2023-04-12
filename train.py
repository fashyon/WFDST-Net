import os
import sys
import cv2
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
from tensorboardX import SummaryWriter
import settings as settings
from dataset  import TrainValDataset,TestDataset
from model import WFDST_Net
from cal_ssim import SSIM

logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
device_ids = range(torch.cuda.device_count())
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)  # +mse
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        ensure_dir(settings.log_test_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if torch.cuda.is_available():
            self.net = WFDST_Net().cuda()
        if len(device_ids) > 1:
            self.net = nn.DataParallel(WFDST_Net()).cuda()
        self.l2 = MSELoss().cuda()
        self.ssim = SSIM().cuda()
        self.epoch = 0
        self.step = 0
        self.ssim_val = 0
        self.psnr_val = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}
        self.opt_net = Adam(self.net.parameters(), lr=5e-4)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1*settings.one_epoch_step, settings.l2*settings.one_epoch_step], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)
        self.writers[name].add_scalar('epoch_step', self.epoch,self.step)
        out['lr'] = self.opt_net.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def get_test_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def save_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_net.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint net%s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint net%s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt_net.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']
        self.sche_net.last_epoch = self.step

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def inf_batch(self, name, batch):
        if name == 'train':
            self.net.zero_grad()
        if self.step == 0:
            self.print_network(self.net)
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        O_derain = self.net(O)
        ssim = self.ssim(O_derain, B)
        loss_O_derain_ssim = -ssim
        loss_O_derain_l2 = self.l2(O_derain, B)
        loss = loss_O_derain_l2 + loss_O_derain_ssim
        if name == 'train':
            loss.backward()
            self.opt_net.step()
        losses = {'loss' : loss}
        ssimes = {'ssim' : ssim}
        losses.update(ssimes)
        self.write(name, losses)
        return O, O_derain, B

    def inf_test_batch(self, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        with torch.no_grad():
            O_derain = self.net(O)
        ssim = self.ssim(O_derain, B)
        loss_ssim = -ssim
        loss_l2 = self.l2(O_derain, B)
        loss = loss_l2 + loss_ssim
        psnr = PSNR(O_derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        loss = loss.data.cpu().numpy()
        return loss, ssim.data.cpu().numpy(), psnr

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data
        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)
        h, w = pred.shape[-2:]
        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

def run_train_val(ckp_name_net='latest_derain_net'):
    sess = Session()
    sess.load_checkpoints_net(ckp_name_net)
    sess.tensorboard('train')
    dt_train = sess.get_dataloader('train')
    while sess.step <= settings.total_step:
        sess.sche_net.step()
        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        O, O_derain, B = sess.inf_batch('train', batch_t)
        sess.epoch = int(sess.step / settings.one_epoch_step)
        if sess.step % (settings.one_epoch_step*50) == 0:
            dt_val = sess.get_test_dataloader('test')
            sess.net.eval()
            ssim_all = 0
            psnr_all = 0
            loss_all = 0
            num_all = 0
            for i, batch_v in enumerate(dt_val):
                print('testing:'+str(i))
                loss, ssim, psnr = sess.inf_test_batch(batch_v)
                ssim_all = ssim_all + ssim
                psnr_all = psnr_all + psnr
                loss_all = loss_all + loss
                num_all = num_all + 1
            loss_avg = loss_all / num_all
            ssim_avg = ssim_all / num_all
            psnr_avg = psnr_all / num_all
            if ssim_avg > sess.ssim_val:
                sess.ssim_val = ssim_avg
                sess.psnr_val = psnr_avg
                sess.save_checkpoints_net('best_net')
            logfile = open(settings.log_test_dir + 'val' + '.txt','a+')
            sess.epoch = int(sess.step / settings.one_epoch_step)
            logfile.write('step  = ' + str(sess.step) + '\t'
                          'epoch = ' + str(sess.epoch) + '\t'
                          'loss  = ' + str(loss_avg) + '\t'
                          'ssim  = ' + str(ssim_avg) + '\t'
                          'psnr  = ' + str(psnr_avg) + '\t'
                          'lr  = ' + str(round(sess.opt_net.param_groups[0]['lr'],7)) + '\t'
                          'batchsize  = ' + str(sess.batch_size) + '\t'
                          'patchsize  = ' + str(settings.patch_size) + '\t''\n\n')
            logfile.close()
        if sess.step % int(sess.save_steps) == 0:
            sess.save_checkpoints_net('latest_net')
        if sess.step % int(settings.one_epoch_step) == 0:
            sess.save_image('train', [batch_t['O'], O_derain, batch_t['B']])
        if sess.step % (settings.one_epoch_step*settings.save_epochs) == 0:
            sess.save_checkpoints_net('step_net_epoch_%d' % sess.epoch)
            logger.info('save model as step_et_epoch_%d' % sess.epoch)
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_net', default='latest_net')
    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model_net)

