import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """Two consecutive conv3x3 → BN → ReLU."""
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
    """MaxPool then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """Bilinear upsample then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x_dec, x_skip):
        x = self.up(x_dec)
        # pad if needed
        dy = x_skip.size(2) - x.size(2)
        dx = x_skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x_skip, x], dim=1))

class VGG19BNBackbone(nn.Module):
    """
    U-Net for density regression using VGG19_bn encoder.
    Outputs [B,1,H/2,W/2] given [B,3,H,W].
    """
    def __init__(self, in_channels=3, pretrained=False):
        super().__init__()
        self.pretrained = pretrained

        if pretrained:
            vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            feats = list(vgg.features.children())
            # Encoder blocks + pools
            self.enc1, self.pool1 = nn.Sequential(*feats[0:6]), feats[6]
            self.enc2, self.pool2 = nn.Sequential(*feats[7:13]), feats[13]
            self.enc3, self.pool3 = nn.Sequential(*feats[14:23]), feats[23]
            self.enc4, self.pool4 = nn.Sequential(*feats[24:33]), feats[33]
            self.enc5, self.pool5 = nn.Sequential(*feats[34:43]), feats[43]
        else:
            # fallback to plain U-Net style
            self.inc   = DoubleConv(in_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 512)

        # Decoder (same for both modes)
        # up1: 512 + 512 → 512
        # up2: 512 + 512 → 256
        # up3: 256 + 256 → 128
        # up4: 128 + 128 → 64   ← note skip from enc2 (128-ch) so output is half-res
        self.up1   = Up(512+512, 512)
        self.up2   = Up(512+512, 256)
        self.up3   = Up(256+256, 128)
        self.up4   = Up(128+128,  64)
        self.outc  = nn.Conv2d(64, 1, kernel_size=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.pretrained:
            x1 = self.enc1(x)
            x2 = self.pool1(x1); x2 = self.enc2(x2)
            x3 = self.pool2(x2); x3 = self.enc3(x3)
            x4 = self.pool3(x3); x4 = self.enc4(x4)
            x5 = self.pool4(x4); x5 = self.enc5(x5)
            x6 = self.pool5(x5)

            d1 = self.up1(x6, x5)
            d2 = self.up2(d1, x4)
            d3 = self.up3(d2, x3)
            d4 = self.up4(d3, x2)   # ← skip from x2 for half-res output
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            d1 = self.up1(x5, x4)
            d2 = self.up2(d1, x3)
            d3 = self.up3(d2, x2)
            d4 = self.up4(d3, x2)   # ← same skip for consistency

        return self.relu(self.outc(d4))
