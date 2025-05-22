import torch.nn as nn
import torchvision.models as models
from .unet_comp import DoubleConv, Down, Up, CustomOutConv


class ResNetUNet(nn.Module):
    """
    ResNet50-based U-Net that matches UNet behavior exactly:
    - Full resolution output (H x W) like UNet
    - Same layer structure as UNet but with ResNet encoder features
    - Variable depth (1-4, matching ResNet layers available)
    - Custom head support
    - No dropout parameter
    """
    def __init__(self, depth: int = 4, custom_head: bool = False, **kwargs):
        super().__init__()
        assert 1 <= depth <= 4, "depth must be between 1 and 4 for ResNet layers"
        self.depth = depth
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Initial convolution WITHOUT pooling (exactly like UNet)
        self.inc = DoubleConv(3, 64, **kwargs)
        
        # Build downsampling path using ResNet layers
        self.downs = nn.ModuleList()
        
        # Manually create Down blocks that use ResNet layers
        if depth >= 1:
            self.downs.append(ResNetDown(64, 256, resnet.layer1, **kwargs))
        if depth >= 2:
            self.downs.append(ResNetDown(256, 512, resnet.layer2, **kwargs))  
        if depth >= 3:
            self.downs.append(ResNetDown(512, 1024, resnet.layer3, **kwargs))
        if depth >= 4:
            self.downs.append(ResNetDown(1024, 2048, resnet.layer4, **kwargs))
        
        # Channel progression
        base_channels = [64, 256, 512, 1024, 2048]
        self.channels = base_channels[:depth+1]
        
        # Bottleneck (exactly like UNet)
        final_ch = self.channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Build ascending path (exactly like UNet)
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch = self.channels[i+1] + self.channels[i]  # decoder + skip
            out_ch = self.channels[i]
            self.ups.append(Up(in_ch, out_ch, **kwargs))
            
        # Output head (exactly like UNet)
        if custom_head:
            self.outc = CustomOutConv(self.channels[0], **kwargs)
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(self.channels[0], 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, return_intermediates=False):
        # Encoder (exactly like UNet structure)
        x_enc = [self.inc(x)]  # [B, 64, H, W]
        
        # Apply downsampling blocks
        for down in self.downs:
            x_enc.append(down(x_enc[-1]))
        
        # Bottleneck
        x_dec = self.bottleneck(x_enc[-1])

        # Decoder (exactly like UNet)
        intermediates = []
        for up, skip in zip(self.ups, reversed(x_enc[:-1])):
            x_dec = up(x_dec, skip)
            if return_intermediates:
                intermediates.append(x_dec)
        
        out = self.outc(x_dec)
        if return_intermediates:
            intermediates.append(out)
            return intermediates
        
        return out


class ResNetDown(nn.Module):
    """Downscaling block that uses ResNet layers instead of DoubleConv"""
    def __init__(self, in_channels, out_channels, resnet_layer, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.resnet_layer = resnet_layer
        
        # Add adapter if input channels don't match ResNet layer expectation
        if hasattr(resnet_layer, '0') and hasattr(resnet_layer[0], 'conv1'):
            expected_in = resnet_layer[0].conv1.in_channels
        else:
            expected_in = in_channels
            
        if in_channels != expected_in:
            self.adapter = nn.Conv2d(in_channels, expected_in, 1)
        else:
            self.adapter = nn.Identity()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.adapter(x)
        return self.resnet_layer(x)


if __name__ == "__main__":
    import torch
    
    # Test the fixed ResNet U-Net
    print("Testing ResNetUNet...")
    model = ResNetUNet(depth=4, custom_head=True)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 1, 256, 256] (full resolution)
    
    # Test with intermediates
    intermediates = model(x, return_intermediates=True)
    print("Number of intermediate outputs:", len(intermediates))
    for i, inter in enumerate(intermediates):
        print(f"Intermediate {i} shape:", inter.shape)
        
    # Compare with regular UNet structure
    print("\nFor comparison - UNet would have these shapes:")
    print("Input: [1, 3, 256, 256]")
    print("After inc: [1, 64, 256, 256]") 
    print("After down1: [1, 256, 128, 128]")
    print("After down2: [1, 512, 64, 64]")
    print("After down3: [1, 1024, 32, 32]")
    print("After down4: [1, 2048, 16, 16]")
    print("Final output: [1, 1, 256, 256]")