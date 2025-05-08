import torch
import torch.nn as nn

class CountingNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),

            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Regressor head for counting
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 8 * 8, 512),  # adjust based on input resolution
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # Mean Squared Error for count regression
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x):
        features = self.features(x)
        count = self.regressor(features)
        return count

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)
