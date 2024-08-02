import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU()
        ])

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x

class Unet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()

        self.input = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512,1024)
        self.convt1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256)
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128)
        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):
        connection = []
        x = self.input(x)
        x = self.down1(x)
        connection.append(x)
        x = self.pool(x)
        x = self.down2(x)
        connection.append(x)
        x = self.pool(x)
        x = self.down3(x)
        connection.append(x)
        x = self.pool(x)
        x = self.down4(x)
        connection.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.convt1(x)
        x = torch.cat([x, connection[-1]], dim=1)
        x = self.up1(x)
        x = self.convt2(x)
        x = torch.cat([x, connection[-2]], dim=1)
        x = self.up2(x)
        x = self.convt3(x)
        x = torch.cat([x, connection[-3]], dim=1)
        x = self.up3(x)
        x = self.convt4(x)
        x = torch.cat([x, connection[-4]], dim=1)
        x = self.up4(x)
        x = self.out(x)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    unet = Unet(3, 10)
    test = unet(torch.rand(10, 3, 128, 128))
    print(test.shape)
