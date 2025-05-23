import torch.nn as nn
import torchvision.models as models

from .unet_comp import Up, CustomOutConv

class ResNetUNet(nn.Module):
    """
    ResNet50-based U-Net with halved-resolution output (H/2 x W/2), and variable depth:
    - depth: number of encoder stages to use (1–5)
    - uses initial conv and first `depth-1` ResNet layers
    - performs (depth-1) ups to return to H/2
    - custom_head: if True, use CustomOutConv; else a 1×1 conv + ReLU
    """
    def __init__(self, depth: int = 5, custom_head: bool = False, **kwargs):
        super().__init__()
        assert 1 <= depth <= 5, "depth must be between 1 and 5"
        self.depth = depth
        self.custom_head = custom_head

        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Encoder modules
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Channel sizes for each encoder output
        self.channels = [64, 256, 512, 1024, 2048]

        # Build decoder: exactly (depth-1) Ups, matching skips
        self.ups = nn.ModuleList()
        for j in range(depth - 1, 0, -1):
            in_ch = self.channels[j]
            skip_ch = self.channels[j - 1]
            self.ups.append(Up(in_ch + skip_ch, skip_ch))

        # Output head
        if custom_head:
            self.outc = CustomOutConv(self.channels[0], **kwargs)
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(self.channels[0], 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, return_intermediates: bool = False):
        # Encoder path
        x1 = self.inc(x)           # [B,64,H/2,W/2]
        x = self.pool(x1)
        x2 = self.layer1(x)        # [B,256,H/4,W/4]
        x3 = self.layer2(x2)       # [B,512,H/8,W/8]
        x4 = self.layer3(x3)       # [B,1024,H/16,W/16]
        x5 = self.layer4(x4)       # [B,2048,H/32,W/32]

        # Collect only up to `depth`
        features = [x1, x2, x3, x4, x5][: self.depth]
        x_dec = features[-1]
        intermediates = []

        # Decoder path: (depth-1) ups, skipping conv1
        skips = features[:-1]
        for up, skip in zip(self.ups, reversed(skips)):
            x_dec = up(x_dec, skip)
            if return_intermediates:
                intermediates.append(x_dec)

        out = self.outc(x_dec)
        if return_intermediates:
            intermediates.append(out)
            return intermediates
        return out
    
if __name__ == "__main__":
    import torch
    # Test the ResNet50 U-Net
    model = ResNetUNet(depth=4, custom_head=True)
    x = torch.randn(1, 3, 256, 256)  # Example input
    output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 1, 128, 128]