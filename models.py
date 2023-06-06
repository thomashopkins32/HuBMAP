import torch
import torch.nn as nn


class UNet2d(nn.Module):
    def __init__(self):
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # TODO: Add batch norm after each conv
        # TODO: Refactor repeated code

        # Input block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        # Input block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, bias=False)
        # Input block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, bias=False)
        # Input block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, bias=False)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, bias=False)
        # Input block 5
        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, bias=False)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, bias=False)

        # Output block 1
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, bias=False)
        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=3, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, bias=False)
        # Output block 2
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, bias=False)
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=3, bias=False)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, bias=False)
        # Output block 3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, bias=False)
        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=3, bias=False)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, bias=False)
        # Output block 4
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, bias=False)
        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=3, bias=False)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        # Final Output
        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, xx):
        # Down
        x1 = self.relu(self.conv1_1(xx))
        x1 = self.relu(self.conv1_2(x1))
        xx = self.max_pool(x1)
        x2 = self.relu(self.conv2_1(xx))
        x2 = self.relu(self.conv2_2(x2))
        xx = self.max_pool(x2)
        x3 = self.relu(self.conv3_1(xx))
        x3 = self.relu(self.conv3_2(x3))
        xx = self.max_pool(x3)
        x4 = self.relu(self.conv4_1(xx))
        x4 = self.relu(self.covn4_2(x4))
        xx = self.max_pool(x4)
        xx = self.relu(self.conv5_1(xx))
        xx = self.relu(self.conv5_2(xx))

        # Up
        xx = self.upconv1(xx)
        xx = torch.cat((x4, xx), dim=1)
        xx = self.relu(self.conv6_1(xx))
        xx = self.relu(self.conv6_2(xx))
        xx = self.upconv2(xx)
        xx = torch.cat((x3, xx), dim=1)
        xx = self.relu(self.conv7_1(xx))
        xx = self.relu(self.conv7_2(xx))
        xx = self.upconv3(xx)
        xx = torch.cat((x2, xx), dim=1)
        xx = self.relu(self.conv8_1(xx))
        xx = self.relu(self.conv8_2(xx))
        xx = self.upconv4(xx)
        xx = torch.cat((x1, xx), dim=1)
        xx = self.relu(self.conv9_1(xx))
        xx = self.relu(self.conv9_2(xx))
        return self.conv_out(xx)




