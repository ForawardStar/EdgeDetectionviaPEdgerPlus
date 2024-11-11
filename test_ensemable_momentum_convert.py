import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models2 import *
from models_noshare import Guider_noshare
from test_datasets import *
from utils import *

import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
from loss_function import *

from pytorch2caffe import pytorch2caffe

def get_model_parm_nums(model): 
    total = sum([param.numel() for param in model.parameters()]) 
    total = float(total) / 1024 
    return total 

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_github/logs/edge_detection/20230827-180330/weights/ckt_0019.pth')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

G_network = Guider_stu()

if cuda:
    G_network = G_network.cuda()
# G_network.
total_params = get_model_parm_nums(G_network)
print("*****************************")
print("total_params:  ", total_params)
print("*****************************")

for name, param in G_network.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=1.3, std=2.5)
        print(name, param.data)
    elif 'bias' in name:
        init.constant_(param,0.7)
        print(name, param.data)

torch.save(G_network.state_dict(), "TestCaffe.pth")

name = 'EdgeRecurrent'
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dummy_input=torch.ones([1,3,481,321]).cuda()
pytorch2caffe.trans_net(G_network, dummy_input, name)
pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))
# ----------
#  Training
# ---------

