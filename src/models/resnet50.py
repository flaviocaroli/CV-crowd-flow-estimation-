import torch.nn as nn
import torchvision.models as models

from .unet_comp import Up


class ResNet50Backbone(nn.Module):
    """
    U-Net for density map regression using ResNet50 encoder.
    Outputs [B,1,H,W] given [B,3,H,W].
    If pretrained=True, uses torchvision ResNet50 encoder with ImageNet weights.
    """

    def __init__(self):
        super().__init__()
        # Load ResNet50 pretrained on ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Initial conv block
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # Encoder layers
        self.down1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64 -> 256
        self.down2 = resnet.layer2  # 256 -> 512, 
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

    def forward(self, x, return_intermediates=False):
        # Encoder path
        x1 = self.inc(x)  # [B,64,H/2,W/2]
        x2 = self.down1(x1)  # [B,256,H/4,W/4]
        x3 = self.down2(x2)  # [B,512,H/8,W/8]
        x4 = self.down3(x3)  # [B,1024,H/16,W/16]
        x5 = self.down4(x4)  # [B,2048,H/32,W/32]
        # Decoder path
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        out = self.outc(d4)
        out = self.relu(out)
        if return_intermediates:
            return [d1, d2, d3, d4, out]
        return out
