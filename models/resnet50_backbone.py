import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def up_block(in_ch, out_ch):
    """
    A small upsampling block: bilinear ×2 → 3×3 conv → BN → ReLU
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ResNet50DensityBackbone(nn.Module):
    """
    U-Net–style ResNet-50 for crowd density estimation.
    Outputs a single-channel density map [B,1,H,W] matching your image inputs.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # 1) Encoder: ResNet50 trunk with dilation in the last stage
        base = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None,
            replace_stride_with_dilation=[False, False, True]
        )

        # Encoder feature stages
        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)  # → [B,  64, H/2,  W/2 ]
        self.pool = base.maxpool                                   # → [B,  64, H/4,  W/4 ]
        self.enc2 = base.layer1                                     # → [B, 256, H/4,  W/4 ]
        self.enc3 = base.layer2                                     # → [B, 512, H/8,  W/8 ]
        self.enc4 = base.layer3                                     # → [B,1024, H/16, W/16]
        self.enc5 = base.layer4                                     # → [B,2048, H/16, W/16]

        # 2) Decoder with skip connections, reversing the reductions
        self.dec5 = up_block(2048, 1024)                        # 16→8
        self.dec4 = up_block(1024 + 512, 512)                   # 8→4
        self.dec3 = up_block(512  + 256, 256)                   # 4→2
        self.dec2 = up_block(256  + 64,  64)                    # 2→1

        # Final 1×1 conv to density
        self.final = nn.Conv2d(64, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: [B, 3, H, W]
          returns density: [B, 1, H, W]
        """
        # --- Encoder ---
        e1 = self.enc1(x)     # [B,  64, H/2,  W/2 ]
        p1 = self.pool(e1)    # [B,  64, H/4,  W/4 ]
        e2 = self.enc2(p1)    # [B, 256, H/4,  W/4 ]
        e3 = self.enc3(e2)    # [B, 512, H/8,  W/8 ]
        e4 = self.enc4(e3)    # [B,1024, H/16, W/16]
        e5 = self.enc5(e4)    # [B,2048, H/16, W/16]

        # --- Decoder + Skips ---
        d5 = self.dec5(e5)                              # [B,1024, H/8, W/8 ]
        d4 = self.dec4(torch.cat([d5, e3], dim=1))      # [B, 512, H/4, W/4 ]
        d3 = self.dec3(torch.cat([d4, e2], dim=1))      # [B, 256, H/2, W/2 ]
        d2 = self.dec2(torch.cat([d3, e1], dim=1))      # [B,  64, H,   W   ]

        # --- Final density map ---
        out = self.final(d2)                            # [B,   1, H,   W   ]
        return out

# Quick sanity check:
if __name__ == "__main__":
    model = ResNet50DensityBackbone(pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
    # Should print: Input: torch.Size([2,3,384,384])  Output: torch.Size([2,1,384,384])
