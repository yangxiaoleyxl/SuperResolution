from torch import nn


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, args["conv1_kernel_num"], kernel_size=args["conv1_kernel_size"],
                               padding=args["conv1_kernel_size"] // 2)
        self.conv2 = nn.Conv2d(args["conv1_kernel_num"], args["conv2_kernel_num"],
                               kernel_size=args["conv2_kernel_size"], padding=args["conv2_kernel_size"] // 2)
        self.conv3 = nn.Conv2d(args["conv2_kernel_num"], 3, kernel_size=args["conv3_kernel_size"],
                               padding=args["conv3_kernel_size"] // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
