import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256), #256x256 could be convolutional
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.resnet(x)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

# Hook into the target conv layer
target_layer = model.layer4[1].conv2  # for example
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)