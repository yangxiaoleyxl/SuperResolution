import math
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, num_channels=3, args=None):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, args["conv1_num"], kernel_size=args["conv1_size"], padding=args["conv1_size"] // 2),
            nn.Tanh(),
            nn.Conv2d(args["conv1_num"], args["conv2_num"], kernel_size=args["conv2_size"],
                      padding=args["conv2_size"] // 2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(args["conv2_num"], num_channels * (args["scale_factor"] ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(args["scale_factor"])
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x
