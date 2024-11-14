import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import torchvision.models as models

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        #self.add_module('norm', nn.BatchNorm2d(out_channel,track_running_stats=False)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class ConvINReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2 + dilation - 1
        super(ConvINReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            #nn.BatchNorm2d(out_channel,track_running_stats=False),
            nn.LeakyReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, dilation):
        super(InvertedResidual, self).__init__()
        expand_ratio = 2.
        hidden_channel = int(in_channel * expand_ratio)

        layers = []
        # 1x1 pointwise conv
        layers.append(ConvINReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvINReLU(hidden_channel, hidden_channel, groups=hidden_channel, dilation=dilation),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, in_channel, kernel_size=1, bias=False),
            #nn.BatchNorm2d(in_channel,track_running_stats=False),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


class Net_NonRecurrent_ResNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Net_NonRecurrent_ResNet50, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.nfc = 32
        self.min_nfc = 32
        self.num_layer = 3
       
        ResNet50 = models.resnet50(pretrained=True)
        
        print("successfully loading pre-trained parameters")

        list_features = list(ResNet50.children())[:6]
        self.resnet50 = nn.Sequential(*list_features)
        self.layer_name_mapping = ['2','3','4','5']

        self.tail_mask_c2f1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_f2c1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_c2f2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_f2c2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_c2f3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_f2c3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_c2f4 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_f2c4 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )


        self.score_final = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        input_curr = x

        mask_features = []
        single_features = []
        state_curr = 0    
        resnet50_outs = []
        
        for name, module in self.resnet50._modules.items():
            x = module(x)
            #print("name:{}  ,  type:{}  ,  x shape:{}".format(name, type(name), x.shape))
            if name in self.layer_name_mapping:
                resnet50_outs.append(x)


        #print("out1:{}  ,  out2:{}  ,  out3:{}  ,  out4:{}".format(vgg16_outs[0].shape, vgg16_outs[1].shape, vgg16_outs[2].shape, vgg16_outs[3].shape))
        prev_features = 0
        for step in range(4):
            state_curr = resnet50_outs[step]
            if step == 0:
                c2f_features1 = self.tail_mask_c2f1(state_curr)
                f2c_features1 = self.tail_mask_f2c1(state_curr)
                #print("c2f_features1:{}  ,  f2c_features1:{}".format(c2f_features1.shape, f2c_features1.shape))
                
                mask_features.append(c2f_features1)
                single_features.append(
                    F.interpolate(f2c_features1, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
                
            elif step == 1:
                c2f_features2 = self.tail_mask_c2f2(state_curr)
                f2c_features2 = self.tail_mask_f2c2(state_curr)
                #print("c2f_features2:{}  ,  f2c_features2:{}".format(c2f_features2.shape, f2c_features2.shape))
                
                features2 = F.interpolate(F.max_pool2d(c2f_features1, kernel_size=2, stride=2, padding=0), size=(c2f_features2.shape[2], c2f_features2.shape[3]), mode='bilinear').detach() + c2f_features2
                 
                mask_features.append(features2)
                single_features.append(
                    F.interpolate(f2c_features2, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
            elif step == 2:
                c2f_features3 = self.tail_mask_c2f3(state_curr)
                f2c_features3 = self.tail_mask_f2c3(state_curr)
                #print("c2f_features3:{}  ,  f2c_features3:{}".format(c2f_features3.shape, f2c_features3.shape))
                
                features3 = features2.detach() + c2f_features3

                mask_features.append(features3)
                single_features.append(
                    F.interpolate(f2c_features3, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
            elif step == 3:
                c2f_features4 = self.tail_mask_c2f4(state_curr)
                f2c_features4 = self.tail_mask_f2c4(state_curr)
                #print("c2f_features4:{}  ,  f2c_features4:{}".format(c2f_features4.shape, f2c_features4.shape))
                
                features4 = F.interpolate(F.max_pool2d(features3, kernel_size=2, stride=2, padding=0), size=(c2f_features4.shape[2], c2f_features4.shape[3]), mode='bilinear').detach() + c2f_features4

                mask_features.append(features4)
                single_features.append(
                    F.interpolate(f2c_features4, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))

        c2f_1 = F.interpolate(mask_features[0], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        c2f_2 = F.interpolate(mask_features[1], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        c2f_3 = F.interpolate(mask_features[2], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        c2f_4 = F.interpolate(mask_features[3], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')

        f2c_1 = single_features[0] + single_features[1].detach() + single_features[2].detach() + single_features[
            3].detach()
        f2c_2 = single_features[1] + single_features[2].detach() + single_features[3].detach()
        f2c_3 = single_features[2] + single_features[3].detach()
        f2c_4 = single_features[3]

        
        fuse = self.score_final(torch.cat([c2f_1, c2f_2, c2f_3, c2f_4, f2c_1, f2c_2, f2c_3, f2c_4], dim=1))
        

        return [c2f_1, c2f_2, c2f_3, c2f_4, f2c_1, f2c_2, f2c_3, f2c_4, fuse]
