import torch
import torch.nn as nn

class SmallRFNet(nn.Module):
    """
    Small receptive field network: uses only 3x3 convolutions and small depth.
    Total parameters: 22,929
    """
    def __init__(self, in_channels=3, base_channels=16):
        super(SmallRFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),  # 448 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),  # 2,320 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # no params
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),  # 4,640 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),  # 9,248 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear((base_channels*2) * (56 // 4) * (56 // 4), 1)  # 6,273 params
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)


class BigRFNet(nn.Module):
    """
    Large receptive field network: uses larger kernels and dilation to expand RF.
    Total parameters: 28,945
    """
    def __init__(self, in_channels=3, base_channels=16):
        super(BigRFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),  # 2,368 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2),  # 6,416 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=2, dilation=2),  # 4,640 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=4, dilation=4),  # 9,248 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear((base_channels*2) * (56 // 4) * (56 // 4), 1)  # 6,273 params
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)


class SmallRFNetGAP(nn.Module):
    """
    Small RF network with Global Average Pooling before regressor.
    Total parameters: 16,689
    """
    def __init__(self, in_channels=3, base_channels=16):
        super(SmallRFNetGAP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),  # 448 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),  # 2,320 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),  # 4,640 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),  # 9,248 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # no params
        self.fc = nn.Linear(base_channels*2, 1)  # 33 params

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class BigRFNetGAP(nn.Module):
    """
    Large RF network with Global Average Pooling before regressor.
    Total parameters: 22,705
    """
    def __init__(self, in_channels=3, base_channels=16):
        super(BigRFNetGAP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),  # 2,368 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2),  # 6,416 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=2, dilation=2),  # 4,640 params
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=4, dilation=4),  # 9,248 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # no params
        self.fc = nn.Linear(base_channels*2, 1)  # 33 params

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)