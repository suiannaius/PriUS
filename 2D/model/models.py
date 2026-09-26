"""PriUS 2D backbone; original parameter names are preserved."""

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet2D, self).__init__()

        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_channels, 1)
        self.softmax = nn.Softmax(dim=out_channels)

    def forward(self, x):
        # print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)

        merge6 = self.offsetCat(c4, up_6)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = self.offsetCat(c3, up_7)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = self.softplus(c10)
        return c10

    def offsetCat(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = down_outputs
        offset = abs(inputs.size()[3] - outputs.size()[3])
        if offset == 1:
            if outputs.device.type == "cuda":
                addition = (
                    torch.rand(
                        (outputs.size()[0], outputs.size()[1], outputs.size()[2]),
                        out=None,
                    )
                    .unsqueeze(3)
                    .cuda()
                )
            else:
                addition = torch.rand(
                    (outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None
                ).unsqueeze(3)
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            if outputs.device.type == "cuda":
                addition = torch.rand(
                    (outputs.size()[0], outputs.size()[1], outputs.size()[2], offset),
                    out=None,
                ).cuda()
            else:
                addition = torch.rand(
                    (outputs.size()[0], outputs.size()[1], outputs.size()[2], offset),
                    out=None,
                )
            outputs = torch.cat([outputs, addition], dim=3)
        out = torch.cat([inputs, outputs], dim=1)
        return out
