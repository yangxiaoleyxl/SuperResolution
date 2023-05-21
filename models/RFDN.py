import torch
import torch.nn as nn
import torch.nn.functional as F


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        distilled_channels = in_channels // 2
        remaining_channels = in_channels
        self.c1_d = nn.Conv2d(in_channels=in_channels, out_channels=distilled_channels, kernel_size=1)
        self.c1_r = nn.Conv2d(in_channels=in_channels, out_channels=remaining_channels, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(in_channels=remaining_channels, out_channels=distilled_channels, kernel_size=1)
        self.c2_r = nn.Conv2d(in_channels=remaining_channels, out_channels=remaining_channels, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(in_channels=remaining_channels, out_channels=distilled_channels, kernel_size=1)
        self.c3_r = nn.Conv2d(in_channels=remaining_channels, out_channels=remaining_channels, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(in_channels=remaining_channels, out_channels=distilled_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = nn.Conv2d(in_channels=distilled_channels * 4, out_channels=in_channels, kernel_size=1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, x):
        distilled_c1 = self.activation(self.c1_d(x))
        r_c1 = (self.c1_r(x))
        r_c1 = self.activation(r_c1 + x)

        distilled_c2 = self.activation(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.activation(r_c2 + r_c1)

        distilled_c3 = self.activation(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.activation(r_c3 + r_c2)

        r_c4 = self.activation(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


class RFDN(nn.Module):
    def __init__(self, num_channel=3, num_features=50, num_modules=4, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = nn.Conv2d(in_channels=num_channel, out_channels=num_features, kernel_size=3, padding=1)

        self.B1 = RFDB(in_channels=num_features)
        self.B2 = RFDB(in_channels=num_features)
        self.B3 = RFDB(in_channels=num_features)
        self.B4 = RFDB(in_channels=num_features)
        self.c = nn.Sequential(
            nn.Conv2d(in_channels=num_features * num_modules, out_channels=num_features, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True))

        self.LR_conv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_channel * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=upscale))
        self.scale_idx = 0

    def forward(self, x):
        out_feature = self.fea_conv(x)
        out_B1 = self.B1(out_feature)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_feature

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
