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

from models_noshare_VGG16 import Net_NonRecurrent_VGG16
from test_datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from loss_function import *

def get_model_parm_nums(model): 
    total = sum([param.numel() for param in model.parameters()]) 
    total = float(total) / 1000 
    return total 

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='models/checkpoint_VGG16.pth')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
opt = parser.parse_args()

# Create sample and checkpoint directories
os.makedirs('images/', exist_ok=True)
os.makedirs('saved_models/', exist_ok=True)

# Losses
criterion = MyLoss()#.cuda()

cuda = True if torch.cuda.is_available() else False

G_network_nonrecurrent1 = Net_NonRecurrent_VGG16()
G_network_nonrecurrent2 = Net_NonRecurrent_VGG16()

if cuda:
    G_network_nonrecurrent1 = G_network_nonrecurrent1.cuda()
    G_network_nonrecurrent2 = G_network_nonrecurrent2.cuda()

# G_network.eval()
# Load pretrained models
if opt.ckpt is not None:
    state_dict = torch.load(opt.ckpt)

    G_network_nonrecurrent1_state_dict1 = state_dict["G_teacher"]
    G_network_nonrecurrent1.load_state_dict(G_network_nonrecurrent1_state_dict1)

    G_network_nonrecurrent2_state_dict2 = state_dict["G_teacher_noshare"]
    G_network_nonrecurrent2.load_state_dict(G_network_nonrecurrent2_state_dict2)
    print("loading checkpoints successfully")

total_params = get_model_parm_nums(G_network_nonrecurrent1)
print("*****************************")
print("total_params:  ", total_params)
print("*****************************")

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
# Image transformations
transforms_ = [transforms.ToTensor(), normalize]
#transforms_ = [transforms.ToTensor()]

# Training data loader
dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, unaligned=True),
                        batch_size=1, shuffle=True, num_workers=1)
# ----------
#  Training
# ---------
prev_time = time.time()

edge_path_formal = "VisualResults_VGG16_ensemble_" + opt.ckpt.strip().strip('/').split('/')[-1]
edge_path_formal1 = "VisualResults_VGG16_nonrecurrent1_" + opt.ckpt.strip().strip('/').split('/')[-1]
edge_path_formal2 = "VisualResults_VGG16_nonrecurrent2_" + opt.ckpt.strip().strip('/').split('/')[-1]
print(edge_path_formal)
os.makedirs(edge_path_formal, exist_ok=True)
os.makedirs(edge_path_formal1, exist_ok=True)
os.makedirs(edge_path_formal2, exist_ok=True)
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input

        #input_edge = Variable(batch['edge'].type(Tensor))
        path_name = batch['path_name'][0]
        file_name = batch['file_name'][0]
        input_image = Variable(batch['img'].type(Tensor))
        
        with torch.no_grad():
            h, w = input_image.shape[2], input_image.shape[3]
            input_image_05 = F.interpolate(input_image, size=(int(round(h*0.5, 0)), int(round(w*0.5, 0))), mode='bilinear')
            input_image_15 = F.interpolate(input_image, size=(int(round(h*1.5, 0)), int(round(w*1.5, 0))), mode='bilinear')

            mask_features_05 = F.interpolate(G_network_nonrecurrent1(input_image_05)[-1], size=(h, w), mode='bilinear')
            mask_features    = G_network_nonrecurrent1(input_image)[-1]
            mask_features_15 = F.interpolate(G_network_nonrecurrent1(input_image_15)[-1], size=(h, w), mode='bilinear')

            mask_features_noshare_05 = F.interpolate(G_network_nonrecurrent2(input_image_05)[-1], size=(h, w), mode='bilinear')
            mask_features_noshare    = G_network_nonrecurrent2(input_image)[-1]
            mask_features_noshare_15 = F.interpolate(G_network_nonrecurrent2(input_image_15)[-1], size=(h, w), mode='bilinear')

            uncertainty_05 = torch.abs(F.sigmoid(mask_features_05) - 0.5).detach()
            uncertainty_noshare_05 = torch.abs(F.sigmoid(mask_features_noshare_05) - 0.5).detach()
            uncertainty = torch.abs(F.sigmoid(mask_features) - 0.5).detach()
            uncertainty_noshare = torch.abs(F.sigmoid(mask_features_noshare) - 0.5).detach()
            uncertainty_15 = torch.abs(F.sigmoid(mask_features_15) - 0.5).detach()
            uncertainty_noshare_15 = torch.abs(F.sigmoid(mask_features_noshare_15) - 0.5).detach()

            weight_05 = uncertainty_05 / (uncertainty_05 + uncertainty_noshare_05)
            weight = uncertainty / (uncertainty + uncertainty_noshare)
            weight_15 = uncertainty_15 / (uncertainty_15 + uncertainty_noshare_15)

            fusion_05 = (mask_features_05 * weight_05 + mask_features_noshare_05 * (1 - weight_05))
            fusion = (mask_features * weight + mask_features_noshare * (1 - weight))
            fusion_15 = (mask_features_15 * weight_15 + mask_features_noshare_15 * (1 - weight_15))

            res_05 = torch.exp(fusion_05.detach() - 0.5) / (torch.exp(fusion_05.detach() - 0.5) + torch.exp(0.5 - fusion_05.detach()))
            res    = torch.exp(fusion.detach() - 0.5) / (torch.exp(fusion.detach() - 0.5) + torch.exp(0.5 - fusion.detach()))
            res_15 = torch.exp(fusion_15.detach() - 0.5) / (torch.exp(fusion_15.detach() - 0.5) + torch.exp(0.5 - fusion_15.detach()))

            res = (res_05 + res + res_15) / 3.0
            res1 = torch.exp(mask_features.detach() - 0.5) / (torch.exp(mask_features.detach() - 0.5) + torch.exp(0.5 - mask_features.detach()))
            res2 = torch.exp(mask_features_noshare.detach() - 0.5) / (torch.exp(mask_features_noshare.detach() - 0.5) + torch.exp(0.5 - mask_features_noshare.detach()))

        
        print(", image_size = {}  , filename = {}".format(res.shape, file_name))
        save_image(res, edge_path_formal + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        save_image(res1, edge_path_formal1 + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        save_image(res2, edge_path_formal2 + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        
        # --------------
        #  Log Progress
        # --------------
        del input_image, path_name, file_name
