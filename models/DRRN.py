from math import sqrt

import torch
import torch.nn as nn


class DRRN(nn.Module):
    def __init__(self, num_channel=3, args=None):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=num_channel, out_channels=args["conv_num"], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=args["conv_num"], out_channels=args["conv_num"], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=args["conv_num"], out_channels=args["conv_num"], kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=args["conv_num"], out_channels=num_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(9):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out
