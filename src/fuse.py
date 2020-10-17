import torch
from utils import *

class FuseBlock(torch.nn.Module):
    def __init__(self, K, C, stride, is_SE, NL, exp, oup):
        super(FuseBlock, self).__init__()
        self.K = K
        self.C = C
        self.stride = stride, 
        self.is_SE = is_SE,
        self.NL = NL
        self.exp = exp
        self.oup = oup
        self.conv1 = torch.nn.Conv2d(
            in_channels = self.C,
            out_channels = self.exp,                   
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        self.bn1 = torch.nn.BatchNorm2d(
            num_features = self.exp
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels = self.exp,
            out_channels = self.exp,
            kernel_size = (1, self.K),
            stride = self.stride,
            padding = (0,int((self.K-1)/2)),
            groups = self.exp,
        )
        self.bn2 = torch.nn.BatchNorm2d(
            num_features = self.exp
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels = self.exp,
            out_channels = self.exp,
            kernel_size = (self.K, 1),
            stride = self.stride,
            padding = (int((self.K-1)/2),0),
            groups = self.exp,
        )
        self.bn3 = torch.nn.BatchNorm2d(
            num_features = self.exp
        )
        self.hsigmoid = Hsigmoid()
        self.SE = SEModule(channel = 2*self.exp)
        self.conv4 = torch.nn.Conv2d(
            in_channels = 2*self.exp,
            out_channels = self.oup,
            stride = 1,
            kernel_size = 1,
            padding = 0
        )
        self.bn4 = torch.nn.BatchNorm2d(
            num_features = self.oup
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.NL(x)
        #print('x')
        #print(x.size())
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x1 = self.bn2(x1)   
        x2 = self.bn3(x2)
        #print('x1')
        #print(x1.size())
        #print('x2')
        #print(x2.size())
        x = torch.cat([x1, x2], 1)
        if self.is_SE:
            x = self.SE(x)
            x = self.hsigmoid(x)
        x = self.NL(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x
