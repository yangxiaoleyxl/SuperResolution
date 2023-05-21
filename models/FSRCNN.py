import math
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, num_channels=3, args=None):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, args['d'], kernel_size=5, padding=5 // 2),
            nn.PReLU(args['d'])
        )
        self.mid_part = [nn.Conv2d(args['d'], args['s'], kernel_size=1), nn.PReLU(args['s'])]
        for _ in range(args['m']):
            self.mid_part.extend([nn.Conv2d(args['s'], args['s'], kernel_size=3, padding=3 // 2), nn.PReLU(args['s'])])
        self.mid_part.extend([nn.Conv2d(args['s'], args['d'], kernel_size=1), nn.PReLU(args['d'])])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(args['d'], num_channels, kernel_size=9, stride=args['scale_factor'],
                                            padding=9 // 2,
                                            output_padding=args["scale_factor"] - 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
