import torch
from torch import nn
from torch.nn import functional as F

class Conv(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = output_channels

        self.convo = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, inputs):
        return self.convo(inputs)
    

class down(nn.Module):
    
    def __init__(self,
                input_channels,
                output_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(stride=2),
            Conv(input_channels=input_channels,
                 output_channels=output_channels)
        )
    
    def forward(self, inputs):
        return self.op(inputs)
    

class upsample(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels):
        super().__init__()
        self.upscale = nn.Upsample(
            mode='bilinear',
            scale_factor=2,
            align_corners=True
        )
        self.conv = Conv(input_channels,
                         output_channels,
                         input_channels/2)
        
    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        d_y = x2.size()[2] - x1.size()[2]
        d_x = x2.size()[3] - x1.size()[3]

        padded_x1 = F.pad(x1,
                          [d_x//2, d_x - (d_x//2),
                           d_y//2, d_y - (d_y//2)])
        
        outputs = torch.cat([x2, x1], dim=1)
        return self.conv(outputs)
    

class output_conv(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=1)
        

    def forward(self, x):
        return self.conv(x)