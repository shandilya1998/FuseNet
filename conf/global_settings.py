import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_TRAIN_STD = (0.2023, 0.1994, 0.2010)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

SAVE_EPOCH = 10
