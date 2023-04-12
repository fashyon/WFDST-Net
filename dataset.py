import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import random
import settings as settings
# import settings15524fjd as settings
from natsort import natsorted


class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = os.path.join(settings.root_dir, name)
        self.root_dir_rain = os.path.join(self.root_dir,"rain")
        self.root_dir_label = os.path.join(self.root_dir, "norain")
        self.mat_files_rain= natsorted(os.listdir(self.root_dir_rain))
        self.mat_files_label= natsorted(os.listdir(self.root_dir_label))
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files_rain)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name_rain = self.mat_files_rain[idx % self.file_num]
        file_name_label = self.mat_files_label[idx % self.file_num]
        img_file_rain = os.path.join(self.root_dir_rain, file_name_rain)
        img_file_label= os.path.join(self.root_dir_label, file_name_label)
        img_rain = cv2.imread(img_file_rain).astype(np.float32) / 255
        img_label = cv2.imread(img_file_label).astype(np.float32) / 255
        if settings.aug_data:
            O, B = self.crop(img_rain,img_label, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            O, B = self.crop(img_rain,img_label, aug=False)
            O, B = self.flip(O, B)
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B}
        return sample

    def crop(self, img_rain,img_label, aug):
        patch_size = self.patch_size
        h, w, c = img_rain.shape
        # w = int(ww / 2)
        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_rain[r: r + p_h, c : c + p_w]
        B = img_label[r: r + p_h, c : c + p_w]
        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))
        return O, B

    def flip(self, O, B):
        tmp=random.choice([0,1, 2, 3])
        if tmp == 1:
            O = cv2.flip(O, 0)
            B = cv2.flip(B, 0)
        if tmp == 2:
            O = cv2.flip(O, 1)
            B = cv2.flip(B, 1)
        if tmp == 3:
            O = cv2.flip(O, -1)
            B = cv2.flip(B, -1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.root_dir = os.path.join(settings.root_dir, name)
        self.root_dir_rain = os.path.join(self.root_dir,"rain")
        self.root_dir_label = os.path.join(self.root_dir, "norain")
        self.mat_files_rain= natsorted(os.listdir(self.root_dir_rain))
        self.mat_files_label= natsorted(os.listdir(self.root_dir_label))
        self.file_num = len(self.mat_files_rain)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name_rain = self.mat_files_rain[idx % self.file_num]
        file_name_label = self.mat_files_label[idx % self.file_num]
        img_file_rain = os.path.join(self.root_dir_rain, file_name_rain)
        img_file_label= os.path.join(self.root_dir_label, file_name_label)
        img_rain = cv2.imread(img_file_rain).astype(np.float32) / 255
        img_label = cv2.imread(img_file_label).astype(np.float32) / 255
        B = np.transpose(img_label, (2, 0, 1))
        O = np.transpose(img_rain, (2, 0, 1))
        sample = {'O': O, 'B': B}
        return sample


class ShowDataset(Dataset):
    def __init__(self,name):
        super().__init__()
        self.root_dir = settings.real_dir
        self.mat_files_rain= natsorted(os.listdir(self.root_dir))
        self.file_num = len(self.mat_files_rain)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files_rain[idx % self.file_num]
        img_file_dir = os.path.join(self.root_dir, file_name)
        img_file = cv2.imread(img_file_dir).astype(np.float32) / 255
        O = np.transpose(img_file, (2, 0, 1))
        sample = {'O': O,  'file_name': file_name}
        return sample


