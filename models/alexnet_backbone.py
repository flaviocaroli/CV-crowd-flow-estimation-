import torch.nn as nn
import torchvision.models as models

class AlexNetBackbone(nn.Module):
    """
    AlexNet backbone + custom 2‑layer head.
    - Uses torchvision.models.alexnet(pretrained=...)
    - Replaces final classifier layer with [Linear→ReLU→Linear(512→1)]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # load pretrained AlexNet
        self.alexnet = models.alexnet(pretrained=pretrained)  # :contentReference[oaicite:2]{index=2}
        # original classifier is Sequential with last layer at index 6
        orig_in = self.alexnet.classifier[6].in_features       # :contentReference[oaicite:3]{index=3}
        # swap in our small head
        # note: nesting a Sequential is fine—AlexNet.forward flattens and then runs classifier
        self.alexnet.classifier[6] = nn.Sequential(
            nn.Linear(orig_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.alexnet(x)
