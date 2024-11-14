import torch.nn as nn
import torch.nn.functional as F
import torch
import time

class SELayer(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel,track_running_stats=False)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class ConvINReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2 + dilation - 1
        super(ConvINReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channel,track_running_stats=False),
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
            nn.BatchNorm2d(in_channel,track_running_stats=False),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


class Net_Recurrent(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Net_Recurrent, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.nfc = 32
        self.min_nfc = 32
        self.num_layer = 3

        self.head = ConvBlock(in_channels, 80, 3, 1, 1)

        self.body = nn.Sequential()

        for i in range(self.num_layer - 2):
            block = InvertedResidual(80, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail_state = nn.Sequential(
            InvertedResidual(80, 1),
            InvertedResidual(80, 1),
            InvertedResidual(80, 1),
            InvertedResidual(80, 2),
            InvertedResidual(80, 2),
            InvertedResidual(80, 4),
            InvertedResidual(80, 4),
            InvertedResidual(80, 8),
            InvertedResidual(80, 8),
            InvertedResidual(80, 1),
            SELayer(80, 80),
        )

        self.tail_mask_c2f = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_f2c = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.channel_expand = nn.Conv2d(1, 80, 1)

        self.score_final = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        input_curr = x

        mask_features = []
        single_features = []
        state_features = []
        state_curr = 0
    
        x1 = self.head(x)
        tail_features = self.body(x1)
        prev_features = 0
        for step in range(5):
            if step > 0:
                recurrent_input = F.max_pool2d(tail_features + state_curr, kernel_size=2, stride=2, padding=0)
            else:
                recurrent_input = tail_features
            state_curr = self.tail_state(recurrent_input)
            c2f_features = self.tail_mask_c2f(state_curr)
            f2c_features = self.tail_mask_f2c(state_curr)
            if step > 0:
                features = F.max_pool2d(prev_features, kernel_size=2, stride=2, padding=0).detach() + c2f_features
            else:
                features = c2f_features

            state_features.append(c2f_features)
            mask_features.append(features)
            single_features.append(
                F.interpolate(f2c_features, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
            tail_features = self.channel_expand(features)
            prev_features = features.detach()

        c2f_1 = mask_features[0]
        c2f_2 = F.interpolate(mask_features[1], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        c2f_3 = F.interpolate(mask_features[2], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        c2f_4 = F.interpolate(mask_features[3], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        c2f_5 = F.interpolate(mask_features[4], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')

        f2c_1 = single_features[0] + single_features[1].detach() + single_features[2].detach() + single_features[
            3].detach() + single_features[4].detach()
        f2c_2 = single_features[1] + single_features[2].detach() + single_features[3].detach() + single_features[4].detach()
        f2c_3 = single_features[2] + single_features[3].detach() + single_features[4].detach()
        f2c_4 = single_features[3] + single_features[4].detach()
        f2c_5 = single_features[4]

        fuse = self.score_final(torch.cat([c2f_1, c2f_2, c2f_3, c2f_4, c2f_5, f2c_1, f2c_2, f2c_3, f2c_4, f2c_5], dim=1))
        
        return [c2f_1, c2f_2, c2f_3, c2f_4, c2f_5, f2c_1, f2c_2, f2c_3, f2c_4, f2c_5, fuse]
