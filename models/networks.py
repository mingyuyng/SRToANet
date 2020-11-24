import torch.nn as nn
from torch.nn import functional as F
from models.layers import *

class DownBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv1d(C_in, C_out, kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.batch = nn.BatchNorm1d(C_out)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.batch(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, p=0, r=2):
        super(UpBlock, self).__init__()
        C_tmp = C_out * r
        self.conv = nn.Conv1d(C_in, C_tmp, kernel_size, padding=(kernel_size-1)//2)
        self.batch = nn.BatchNorm1d(C_tmp)
        self.relu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p)
        self.pixelshuffle = SubPixel1D(r)

    def forward(self, x, x_up=None):
        x = self.conv(x)
        x = self.batch(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.pixelshuffle(x)
        
        if x_up is None:
            return x
        else:
            return torch.cat((x, x_up), 1)


class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        self.down1 = DownBlock(2, 64, 9)
        self.down2 = DownBlock(64, 128, 9)
        self.down3 = DownBlock(128, 256, 5)
        self.down4 = DownBlock(256, 512, 5)

        self.bottleneck = DownBlock(512, 512, 5)
        
        self.up1 = UpBlock(512, 512, 5, r=2)
        self.up2 = UpBlock(1024, 256, 5, r=2)
        self.up3 = UpBlock(512, 128, 9, r=2)
        self.up4 = UpBlock(256, 64, 9, r=2)

        self.final = UpBlock(128, 2, 9, r=2)

    def forward(self, x):
        
        x1 = self.down1(x)        
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.bottleneck(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        x_out = self.final(x9)

        return x_out+x


class Flatten(nn.Module):
  def forward(self, x):
    N, C, H = x.size() # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Sandwich(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super(Sandwich, self).__init__()
        self.conv = nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.batch = nn.BatchNorm1d(C_out)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.batch(self.conv(x)))

class Regnet(nn.Module):
    def __init__(self, len_in, C_in):
        super(Regnet, self).__init__()

        self.net = nn.Sequential(Sandwich(C_in, 64, 9, stride=2),
                       Sandwich(64, 128, 5, stride=2),
                       Sandwich(128, 256, 3, stride=2),
                       Flatten(),
                       nn.Linear(256*len_in//8, 1024),
                       nn.ReLU(True),
                       nn.Linear(1024,256),
                       nn.ReLU(True),
                       nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)