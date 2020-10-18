import torch
from src.utils import Hswish
from src.fuse import FuseBlock

class FuseNet(torch.nn.Module):
    def __init__(self, H, W, C):
        super(FuseNet, self).__init__()
        self.relu = torch.nn.ReLU()
        self.hswish = Hswish()
        self.conv1 = torch.nn.Conv2d(
            in_channels = C,
            out_channels = 16,
            stride = 2, 
            kernel_size = 3, 
            padding = 1 ,
        )
        self.fuse1 = FuseBlock(
            K = 3,
            C = 16,
            stride = 2,
            is_SE = True, 
            NL = self.relu,
            exp = 16, 
            oup = 16
        )

        self.fuse2 = FuseBlock(
            K = 3,
            C = 16,
            stride = 2,
            is_SE = False, 
            NL = self.relu,
            exp = 72, 
            oup = 24
        ) 

        self.fuse3 = FuseBlock(
            K = 3,
            C = 24,
            stride = 1,
            is_SE = False, 
            NL = self.relu,
            exp = 88, 
            oup = 24
        ) 
        
        self.fuse4 = FuseBlock(
            K = 5,
            C = 24,
            stride = 2,
            is_SE = True, 
            NL = self.hswish,
            exp = 96, 
            oup = 40
        ) 

        self.fuse5 = FuseBlock(
            K = 5,
            C = 40,
            stride = 1,
            is_SE = True, 
            NL = self.hswish,
            exp = 240,
            oup = 40
        )

        self.fuse6 = FuseBlock(
            K = 5,
            C = 40,
            stride = 1,
            is_SE = True,
            NL = self.hswish,
            exp = 240,
            oup = 40
        )
        
        self.fuse7 = FuseBlock(
            K = 5,
            C = 40,
            stride = 1,
            is_SE = True,
            NL = self.hswish,
            exp = 120,
            oup = 48
        )
        
        self.fuse8 = FuseBlock(
            K = 5, 
            C = 48, 
            stride = 1, 
            is_SE = True, 
            NL = self.hswish,
            exp = 144, 
            oup = 48
        )

        self.fuse9 = FuseBlock(
            K = 5, 
            C = 48, 
            stride = 2, 
            is_SE = True, 
            NL = self.hswish,
            exp = 288,
            oup = 96
        )

        self.fuse10 = FuseBlock(
            K = 5,
            C = 96, 
            stride = 1, 
            is_SE = True, 
            NL = self.hswish,
            exp = 576, 
            oup = 96, 
        )

        self.fuse11 = FuseBlock(
            K = 5, 
            C = 96, 
            stride = 1, 
            is_SE = True, 
            NL = self.hswish, 
            exp = 576, 
            oup = 96, 
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels = 96,
            out_channels = 576,
            stride = 1,
            kernel_size = 1,
            padding = 0,
        )

        self.pool = torch.nn.AdaptiveAvgPool2d(
            output_size = (1, 1)
        )
        
        self.conv3 = torch.nn.Conv2d(
            in_channels = 576,
            out_channels = 1024,
            stride = 1,
            kernel_size = 1,
            padding = 0,
        )

        self.dropout = torch.nn.Dropout2d(p=0.2)    

        self.conv4 = torch.nn.Conv2d(
            in_channels = 1024,
            out_channels = 100,
            stride = 1,
            kernel_size = 1,
            padding = 0,
        )

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.hswish(x)
        x = self.fuse1(x)
        x = self.fuse2(x)
        x = self.fuse3(x)
        x = self.fuse4(x)
        x = self.fuse5(x)
        x = self.fuse6(x)
        x = self.fuse7(x)
        x = self.fuse8(x)
        x = self.fuse9(x)
        x = self.fuse10(x)
        x = self.fuse11(x) 
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        b = x.size()[0]
        x = torch.reshape(x, (b, 100))
        return x
