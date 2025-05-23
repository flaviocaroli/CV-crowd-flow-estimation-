import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive conv3x3 -> BN -> ReLU, with optional Dropout
    """

    def __init__(self, in_ch:int, out_ch:int, **kwargs):
        super().__init__()
        dropout:float = kwargs.get("dropout", 0.0)
        stride_l1:int = kwargs.get("stride_l1", 1)
        stride_l2:int = kwargs.get("stride_l2", 1)
        dilation_l1:int = kwargs.get("dilation_l1", 1)
        dilation_l2:int = kwargs.get("dilation_l2", 1)
        # first conv block
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
                stride=stride_l1,
                dilation=dilation_l1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
                stride=stride_l2,
                dilation=dilation_l2,
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), 
                                  DoubleConv(in_ch, out_ch, **kwargs))

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        mode = kwargs.get("upsampling_mode", "bilinear")

        # use bilinear upsampling
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CustomOutConv(nn.Module):
    """
    Final convolution to map to 1 output channel.
    Optionally applies Dropout and Global Average Pooling.
    """

    def __init__(self, F, **kwargs):
        super().__init__()

        kernel_size = kwargs.get("custom_head_kernel_size", 3)
        dropout_p = kwargs.get("custom_head_dropout", 0.0)
        gap = kwargs.get("custom_head_gap", False)

        layers = []
        # first conv block
        layers.append(
            nn.Conv2d(
                F, F // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
            )
        )
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        layers.append(nn.BatchNorm2d(F // 2))
        layers.append(nn.LeakyReLU(inplace=True))

        # second conv to single channel
        layers.append(nn.Conv2d(F // 2, 1, kernel_size=1))
        layers.append(nn.LeakyReLU(inplace=True))

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
