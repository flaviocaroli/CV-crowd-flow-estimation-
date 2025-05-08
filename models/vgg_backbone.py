import torch.nn as nn
import torchvision.models as models


class VGG19BNBackbone(nn.Module):
    """
    VGG‑19 with BatchNorm backbone + custom 2‑layer head.
    - Uses torchvision.models.vgg19_bn(pretrained=...)
    - Replaces classifier with [Linear→ReLU→Linear(512→1)]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # load VGG‑19 with batch‑norm
        self.vgg = models.vgg19_bn(pretrained=pretrained)  # :contentReference[oaicite:0]{index=0}
        # grab the in_features of the original last fc
        in_feats = self.vgg.classifier[6].in_features         # :contentReference[oaicite:1]{index=1}
        # replace entire classifier
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.vgg(x)
