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

from models import *
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
parser.add_argument('--ckpt', type=str, default='models/checkpoint.pth')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
opt = parser.parse_args()

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion = MyLoss()#.cuda()

cuda = True if torch.cuda.is_available() else False

G_network = Guider_stu()
G_network_noshare = Guider_noshare()

if cuda:
    G_network = G_network.cuda()
    G_network_noshare = G_network_noshare.cuda()

# G_network.eval()
# Load pretrained models
if opt.ckpt is not None:
    state_dict = torch.load(opt.ckpt)

    #G_network_state_dict = state_dict["G_teacher"]
    #G_network.load_state_dict(G_network_state_dict)

    #G_network_noshare_state_dict = state_dict["G_teacher_noshare"]
    #G_network_noshare.load_state_dict(G_network_noshare_state_dict)
    G_network_state_dict = state_dict["G_teacher"]
    G_network.load_state_dict(G_network_state_dict)

    G_network_noshare_state_dict = state_dict["G_teacher_noshare"]
    G_network_noshare.load_state_dict(G_network_noshare_state_dict)

total_params = get_model_parm_nums(G_network)
print("*****************************")
print("total_params: {} KB".format(total_params))
print("*****************************")

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [transforms.ToTensor()]

# Training data loader
dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, unaligned=True),
                        batch_size=1, shuffle=True, num_workers=1)
# ----------
#  Training
# ---------
prev_time = time.time()

edge_path_formal = "VisualResults_" + opt.ckpt.strip().strip('/').split('/')[-1].replace(".pth", "")
print(edge_path_formal)
os.makedirs(edge_path_formal, exist_ok=True)
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input

        #input_edge = Variable(batch['edge'].type(Tensor))
        path_name = batch['path_name'][0]
        file_name = batch['file_name'][0]
        input_image = Variable(batch['img'].type(Tensor))
        _input_image = input_image.clone()
        
        with torch.no_grad():
            h, w = input_image.shape[2], input_image.shape[3]
            
            mask_features    = G_network(input_image)[-1]
            
            res = torch.exp(mask_features.detach() - 0.5) / (torch.exp(mask_features.detach() - 0.5) + torch.exp(0.5 - mask_features.detach()))
            
        # print("head.norm.running_mean[0] = ", G_network.state_dict()["head.norm.running_mean"][0].item(), end=' ')
        #outputs = [torch.sigmoid(r) for r in outputs]

        #res = torch.exp(mask_features.detach() - 0.5) / (torch.exp(mask_features.detach() - 0.5) + torch.exp(0.5 - mask_features.detach()))
        print(", image_size = ", res.shape)
        save_image(res, edge_path_formal + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        
        # --------------
        #  Log Progress
        # --------------
        del input_image, path_name, file_name
