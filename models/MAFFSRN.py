import torch
import torch.nn.functional as F
from torch import nn


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class MAB(nn.Module):
    def __init__(self, num_features, reduction_factor=4, distillation_rate=0.25):
        super(MAB, self).__init__()
        reduced_features = num_features // reduction_factor
        self.reduce_channels = nn.Conv2d(in_channels=num_features, out_channels=reduced_features, kernel_size=1)
        self.reduce_spatial_size = nn.Conv2d(in_channels=reduced_features, out_channels=reduced_features, kernel_size=3,
                                             stride=2, padding=1)
        self.pool = nn.MaxPool2d(7, stride=3)
        self.increase_channels = nn.Conv2d(in_channels=reduced_features, out_channels=num_features, kernel_size=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_features, out_channels=reduced_features, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_features, out_channels=reduced_features, kernel_size=3, padding=2,
                      dilation=2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True))

        self.sigmoid = nn.Sigmoid()

        self.conv00 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.conv01 = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True))

        self.bottom11 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1)
        self.bottom11_dw = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=5, padding=2,
                                     groups=num_features)

    def forward(self, x):
        x = self.conv01(x)
        x = self.conv00(x)
        rc = self.reduce_channels(x)
        rs = self.reduce_spatial_size(rc)
        pool = self.pool(rs)
        conv = self.conv2(pool)
        conv = conv + self.conv1(pool)
        up = torch.nn.functional.upsample(conv, size=(rc.shape[2], rc.shape[3]), mode='nearest')
        up = up + rc
        out = (self.sigmoid(self.increase_channels(up)) * x) * self.sigmoid(self.bottom11_dw(self.bottom11(x)))
        return out


class FFG(nn.Module):
    def __init__(self, num_features, wn):
        super(FFG, self).__init__()

        self.b0 = MAB(num_features=num_features, reduction_factor=4)
        self.b1 = MAB(num_features=num_features, reduction_factor=4)
        self.b2 = MAB(num_features=num_features, reduction_factor=4)
        self.b3 = MAB(num_features=num_features, reduction_factor=4)

        self.reduction1 = wn(nn.Conv2d(in_channels=num_features * 2, out_channels=num_features, kernel_size=1))
        self.reduction2 = wn(nn.Conv2d(in_channels=num_features * 2, out_channels=num_features, kernel_size=1))
        self.reduction3 = wn(nn.Conv2d(in_channels=num_features * 2, out_channels=num_features, kernel_size=1))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0) + x0
        x2 = self.b2(x1) + x1
        x3 = self.b3(x2)

        res1 = self.reduction1(channel_shuffle(torch.cat([x0, x1], dim=1), 2))
        res2 = self.reduction2(channel_shuffle(torch.cat([res1, x2], dim=1), 2))
        res = self.reduction3(channel_shuffle(torch.cat([res2, x3], dim=1), 2))

        return self.res_scale(res) + self.x_scale(x)


class Tail(nn.Module):
    def __init__(self, args, scale, num_features, wn):
        super(Tail, self).__init__()
        out_features = scale * scale * 3
        self.tail_k3 = wn(nn.Conv2d(in_channels=num_features, out_channels=out_features, kernel_size=3, padding=1))
        self.tail_k5 = wn(nn.Conv2d(in_channels=num_features, out_channels=out_features, kernel_size=5, padding=2))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))

        return x0 + x1


class MAFFSRN(nn.Module):
    def __init__(self, args):
        super(MAFFSRN, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.scale = args['scale_factor']
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # define head module
        head = [wn(nn.Conv2d(in_channels=3, out_channels=args["num_features"], kernel_size=3, padding=1))]

        # define body module
        body = []
        for i in range(args["num_FFGs"]):
            body.append(FFG(args["num_features"], wn=wn))

        # define tail module
        tail = Tail(args, self.scale, args["num_features"], wn)

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail

    def forward(self, x):
        x0 = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + torch.nn.functional.upsample(x0, scale_factor=self.scale, mode='bicubic')

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
