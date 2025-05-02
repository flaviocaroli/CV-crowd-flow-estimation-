import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.resnet(x)