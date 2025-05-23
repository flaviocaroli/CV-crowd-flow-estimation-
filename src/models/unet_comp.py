import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two consecutive conv3x3 -> BN -> ReLU, with optional Dropout
    """
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        dropout = kwargs.get('dropout', 0.0)
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 **kwargs):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), # downsample by 2
            DoubleConv(in_ch, out_ch, **kwargs) # pass through double conv 
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, 
                 in_ch, 
                 out_ch,
                 **kwargs):
        super().__init__()
        mode =  kwargs.get("upsampling_mode", "bilinear") 

        # use bilinear upsampling
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False) # upsample by 2
        self.conv = DoubleConv(in_ch, out_ch) # pass through double conv

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if necessary. Can happen if the input size is not divisible by 2^depth e.g
        # e.g. if no padding in the conv layers
        # IN OUR CASE, THIS SHOULD NOT HAPPEN
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CustomOutConv(nn.Module):
    """
    Final convolution to map to 1 output channel.
    Optionally applies Dropout and Global Average Pooling.
    """
    def __init__(self, F, **kwargs):
        super().__init__()

        kernel_size = kwargs.get('custom_head_kernel_size', 3)
        dropout_p  = kwargs.get('custom_head_dropout', 0.0)
        gap        = kwargs.get('custom_head_gap', False)

        layers = []
        # first conv block
        layers.append(nn.Conv2d(F, F//2,
                                kernel_size=kernel_size,
                                padding=kernel_size//2,
                                bias=False))
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        layers.append(nn.BatchNorm2d(F//2))
        layers.append(nn.ReLU(inplace=True))

        # second conv to single channel
        layers.append(nn.Conv2d(F//2, 1, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        # optional global average pooling
        if gap:
            layers.append(nn.AdaptiveAvgPool2d(1))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Tensor of shape (B, F, H, W)
        returns:
          if gap=False -> (B, 1, H, W)
          if gap=True  -> (B, 1, 1, 1)
        """
        return self.head(x)