import torch
import torch.nn as nn

from utils import *


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        Note: We pad the convolutions here since we know the input image size is always 512x512.
        Otherwise we would do the "Overlap-tile strategy" presented in the u-net paper.
        '''
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, xx):
        return self.model(xx)


class UNet2d(nn.Module):
    ''' Paper: https://arxiv.org/pdf/1505.04597v1.pdf '''
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Down
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)

        # Up
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(128, 64)
        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, xx):
        # Down
        x1 = self.conv1(xx)
        xx = self.max_pool(x1)
        x2 = self.conv2(xx)
        xx = self.max_pool(x2)
        x3 = self.conv3(xx)
        xx = self.max_pool(x3)
        x4 = self.conv4(xx)
        xx = self.max_pool(x4)
        xx = self.conv5(xx)

        # Up
        xx = self.upconv1(xx)
        xx = torch.cat((x4, xx), dim=1)
        xx = self.conv6(xx)
        xx = self.upconv2(xx)
        xx = torch.cat((x3, xx), dim=1)
        xx = self.conv7(xx)
        xx = self.upconv3(xx)
        xx = torch.cat((x2, xx), dim=1)
        xx = self.conv8(xx)
        xx = self.upconv4(xx)
        xx = torch.cat((x1, xx), dim=1)
        xx = self.conv9(xx)
        return self.conv_out(xx)


if __name__ == '__main__':
    model = UNet2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    memory_usage_stats(model, optimizer, batch_size=4, device='cuda')
