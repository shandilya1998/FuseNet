import torch

class Hsigmoid(torch.nn.Module):
    def __init__(self, inplace = True):
        super(Hsigmoid, self).__init__() 
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.relu(x+3, inplace = self.inplace)/6.0

class SEModule(torch.nn.Module):
    def __init__(self, channel, reduction = 4):
        super(SEModule, self).__init__() 
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel//reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 

class Hswish(torch.nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3., inplace=self.inplace) / 6.   
