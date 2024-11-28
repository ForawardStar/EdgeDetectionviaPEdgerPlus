from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
import time


class _Base(nn.Module):
    _version = 2
    __constants__ = ["momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_Base, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_Base, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class Bayesian_Samp(_Base):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.9,
            affine: bool = True,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bayesian_Samp, self).__init__(
            num_features, eps, momentum, affine, **factory_kwargs
        )

    def bayesian_sampling(self, input, weight, bias, eps=1e-5):
        reduced_dim = [i for i in range(input.dim()) if i not in [0, 1]]
        normalized_shape = [1] * len(input.shape)
        normalized_shape[1] = input.shape[1]

        shape = [1] * len(input.shape)
        shape[:2] = input.shape[:2]

        mean = input.mean(dim=reduced_dim)
        var = input.var(dim=reduced_dim, unbiased=False)
        mean_update = mean.mean(0)
        var_update = input.var(dim=reduced_dim, unbiased=True).mean(0)

        x_hat = (input - mean.view(*shape)) / torch.sqrt(var.view(*shape) + eps)

        x = x_hat * weight.view(*normalized_shape) + bias.view(*normalized_shape)
        return x

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        return self.bayesian_sampling(
            input,
            self.weight,
            self.bias
        )

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


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
        self.add_module('norm', Bayesian_Samp(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class ConvINReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2 + dilation - 1
        super(ConvINReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            Bayesian_Samp(out_channel),
            nn.LeakyReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, dilation):
        super(InvertedResidual, self).__init__()
        expand_ratio = 0.8
        hidden_channel = int(in_channel * expand_ratio)

        layers = []
        # 1x1 pointwise conv
        layers.append(ConvINReLU(in_channel, hidden_channel, kernel_size=3))
        layers.extend([
            nn.Conv2d(hidden_channel, in_channel, kernel_size=1, bias=False),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


class Net_Recurrent_bayesian(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Net_Recurrent_bayesian, self).__init__()
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

        self.tail_mask_s2d = nn.Sequential(
            nn.Conv2d(80, 16, kernel_size=3, stride=1, padding=1),
            Bayesian_Samp(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_d2s = nn.Sequential(
            nn.Conv2d(80, 16, kernel_size=3, stride=1, padding=1),
            Bayesian_Samp(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.channel_expand = nn.Conv2d(1, 80, 1)

        self.score_final = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        input_curr = x

        mask_features = []
        single_features = []
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
            s2d_features = self.tail_mask_s2d(state_curr)
            d2s_features = self.tail_mask_d2s(state_curr)
            if step > 0:
                features = F.max_pool2d(prev_features, kernel_size=2, stride=2, padding=0).detach() + s2d_features
            else:
                features = s2d_features

            mask_features.append(features)
            single_features.append(
                F.interpolate(d2s_features, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
            tail_features = self.channel_expand(features)
            prev_features = features.detach()

        s2d_1 = mask_features[0]
        s2d_2 = F.interpolate(mask_features[1], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        s2d_3 = F.interpolate(mask_features[2], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        s2d_4 = F.interpolate(mask_features[3], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        s2d_5 = F.interpolate(mask_features[4], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')

        d2s_1 = single_features[0] + single_features[1].detach() + single_features[2].detach() + single_features[
            3].detach() + single_features[4].detach()
        d2s_2 = single_features[1] + single_features[2].detach() + single_features[3].detach() + single_features[4].detach()
        d2s_3 = single_features[2] + single_features[3].detach() + single_features[4].detach()
        d2s_4 = single_features[3] + single_features[4].detach()
        d2s_5 = single_features[4]

        fuse = self.score_final(torch.cat([s2d_1, s2d_2, s2d_3, s2d_4, s2d_5, d2s_1, d2s_2, d2s_3, d2s_4, d2s_5], dim=1))
        
        return [s2d_1, s2d_2, s2d_3, s2d_4, s2d_5, d2s_1, d2s_2, d2s_3, d2s_4, d2s_5, fuse]
