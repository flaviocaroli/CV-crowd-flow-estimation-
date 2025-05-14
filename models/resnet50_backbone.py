import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """
    Two consecutive conv3x3 -> BN -> ReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # use bilinear upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResNet50Backbone(nn.Module):
    """
    U-Net for density map regression using ResNet50 encoder.
    Outputs [B,1,H,W] given [B,3,H,W].
    If pretrained=True, uses torchvision ResNet50 encoder with ImageNet weights.
    """
    def __init__(self, in_channels=3, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        if pretrained:
            # Load ResNet50 pretrained on ImageNet
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Initial conv block
            self.inc = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            )
            # Encoder layers
            self.down1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64 -> 256
            self.down2 = resnet.layer2  # 256 -> 512
            self.down3 = resnet.layer3  # 512 -> 1024
            self.down4 = resnet.layer4  # 1024 -> 2048
            # Decoder layers (with skip connections)
            self.up1 = Up(2048 + 1024, 1024)
            self.up2 = Up(1024 + 512, 512)
            self.up3 = Up(512 + 256, 256)
            self.up4 = Up(256 + 64, 64)
            # Final conv to density map
            self.outc = nn.Conv2d(64, 1, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
        else:
            # Standard U-Net encoder
            self.inc = DoubleConv(in_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 512)
            # Decoder
            self.up1 = Up(512 + 512, 256)
            self.up2 = Up(256 + 256, 128)
            self.up3 = Up(128 + 128, 64)
            self.up4 = Up(64 + 64, 64)
            self.outc = nn.Conv2d(64, 1, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.pretrained:
            # Encoder path
            x1 = self.inc(x)      # [B,64,H/2,W/2]
            x2 = self.down1(x1)   # [B,256,H/4,W/4]
            x3 = self.down2(x2)   # [B,512,H/8,W/8]
            x4 = self.down3(x3)   # [B,1024,H/16,W/16]
            x5 = self.down4(x4)   # [B,2048,H/32,W/32]
            # Decoder path
            d1 = self.up1(x5, x4)
            d2 = self.up2(d1, x3)
            d3 = self.up3(d2, x2)
            d4 = self.up4(d3, x1)
            out = self.outc(d4)
            return self.relu(out)
        # Non-pretrained forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        out = self.outc(d4)
        return self.relu(out)
