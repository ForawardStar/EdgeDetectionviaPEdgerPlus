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
import torch.nn.functional as F
import torch
from loss_function import *

def get_model_parm_nums(model): 
    total = sum([param.numel() for param in model.parameters()]) 
    total = float(total) / 1024 
    return total 

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='TestCaffe.pth')
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

# Losses
criterion = MyLoss()#.cuda()

cuda = True if torch.cuda.is_available() else False

G_network = Guider_stu()

if cuda:
    G_network = G_network.cuda()

# G_network.eval()
# Load pretrained models
if opt.ckpt is not None:
    state_dict = torch.load(opt.ckpt)

    G_network.load_state_dict(state_dict)

total_params = get_model_parm_nums(G_network)
print("*****************************")
print("total_params:  ", total_params)
print("*****************************")

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
# Image transformations
transforms_ = [transforms.ToTensor(), normalize]
dataloader = DataLoader(ImageDataset("/home/fyb", transforms_=transforms_, unaligned=True),
                        batch_size=1, shuffle=True, num_workers=1)
prev_time = time.time()

edge_path_formal = opt.ckpt.strip().strip('/').split('/')[-1].replace(".pth", "")
print(edge_path_formal)
os.makedirs(edge_path_formal, exist_ok=True)

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        path_name = batch['path_name'][0]
        file_name = batch['file_name'][0]
        input_image = Variable(batch['img'].type(Tensor))
        _input_image = input_image.clone()
        
        with torch.no_grad():
            h, w = input_image.shape[2], input_image.shape[3]
            
            mask_features    = G_network(input_image)[-1]
            print(mask_features)
            res = (mask_features - torch.min(mask_features)) / (torch.max(mask_features) - torch.min(mask_features))
        
        save_image(res, edge_path_formal + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        
        del input_image, path_name, file_name
