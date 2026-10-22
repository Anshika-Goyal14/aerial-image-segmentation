import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1, bias = False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True), #bias = false removes unncessary parameters 
                            nn.Sequential(nn.Conv2d(out_channels, out_channels, 3,padding= 1, bias = False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True) ))

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor =2, mode = 'bilinear', align_corners = True) #bilinear upsample ismore stable than Convtranspose2d

        self.conv = DoubleConv(in_channels, out_channels)

    def forward (self, x, skip):
        x = self.upsample(x)
        x = torch.cat([skip,x], dim =1) #concatinating along dim
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 5):
        super().__init__()

        #encoder
        self.input = DoubleConv(in_channels, 64) # (3 to 64, 512 to 512)
        self.down1 = DownSample(64,128)
        self.down2 = DownSample(128,256)
        self.down3 = DownSample(256,512)
        self.down4 = DownSample(512,1024)

        #decoder 
        self.up1 = UpSample(1024+512, 512)
        self.up2 = UpSample(512+256, 256)
        self.up3 = UpSample(256+128, 128)
        self.up4 = UpSample(128+64, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1) #1x1 map , num_classes is per pixel for 64 channels, maps them to 5 classes 

    def forward(self,x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  #bottleneck 

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.output(x)

if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=5)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(out.shape) 
    
    












