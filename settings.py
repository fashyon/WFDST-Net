import os
import logging

root_dir = './datasets/Rain200H'
real_dir = './datasets/real'
log_dir = './logdir'
log_test_dir = './log_test/'
show_dir = './showdir'
model_dir = './models'
data_dir = os.path.join(root_dir, 'train/rain')
mat_files = os.listdir(data_dir)
num_datasets = len(mat_files)

channel = 64
window_size = 8
depth = 2
heads = 2


lr = 5e-4
batch_size = 8
patch_size = 128
epoch = 500
aug_data = False
l1 = 400
l2 = 450
total_step = int((epoch * num_datasets)/batch_size)
one_epoch_step = int(num_datasets/batch_size)
save_steps = 50
save_epochs = 50

num_workers = 0
num_GPU = 1
device_id = ''
for i in range(num_GPU):
    device_id += str(i) + ','


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


