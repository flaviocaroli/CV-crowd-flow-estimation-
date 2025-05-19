import torch.nn as nn
from .unet_comp import DoubleConv, Down, Up

class UNet4(nn.Module):
    """
    U-Net for density map regression using ResNet50 encoder.
    Outputs [B,1,H,W] given [B,3,H,W].
    If pretrained=True, uses torchvision ResNet50 encoder with ImageNet weights.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)
        # Decoder
        self.up1 = Up(base_channels * 8 + base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4 + base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2 + base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, 
                x, 
                return_intermediates=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        out = self.outc(d3)
        out = self.relu(out)
        if return_intermediates:
            return [d1, d2, d3, out]
        return out
