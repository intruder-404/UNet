from base import *

class UNet(nn.Module):

    def __init__(self,
                 n_channels,
                 n_classes):
        super().__init__()

        self.n_channels, self.n_classes = (n_channels, n_classes)
        self.entry = Conv(n_channels, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = upsample(1024, 256)
        self.up2 = upsample(512, 128)
        self.up3 = upsample(256, 64)
        self.up4 = upsample(128, 64)

        self.out = output_conv(64, n_classes)


    def forward(self, x):
        _x = self.entry(x)
        x1 = self.down1(_x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, _x)
        return self.out(x)