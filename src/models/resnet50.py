import torch.nn as nn
import torchvision.models as models
from .unet_comp import DoubleConv, Up, CustomOutConv


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
        self.bottleneck = DoubleConv(final_ch, final_ch, **kwargs)

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


class ResnetSkip1(nn.Module):
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
    
    # Test the original ResNet U-Net
    model_kwargs = {
        "dilation_l1": 1,
        "dilation_l2": 1,
    }
    print("Testing ResNetUNet (full resolution)...")
    model_full = ResNetUNet(depth=4, custom_head=True, half_resolution=False, **model_kwargs)
    x = torch.randn(1, 3, 256, 256)
    output_full = model_full(x)
    print("Full resolution output shape:", output_full.shape)  # Should be [1, 1, 256, 256]
    
    # Test the half resolution ResNet U-Net
    print("\nTesting ResNetUNet (half resolution)...")
    model_half = ResNetUNet(depth=4, custom_head=True, half_resolution=True, **model_kwargs)
    output_half = model_half(x)
    print("Half resolution output shape:", output_half.shape)  # Should be [1, 1, 128, 128]
    
    # Test with intermediates for half resolution
    intermediates = model_half(x, return_intermediates=True )
    print("Number of intermediate outputs (half res):", len(intermediates))
    for i, inter in enumerate(intermediates):
        print(f"Intermediate {i} shape:", inter.shape)
        
    print("\nHalf resolution model architecture:")
    print("- Skips the last decoder layer")
    print("- Skip connections start from second downsampling layer")
    print("- Output resolution is H/2 x W/2 instead of H x W")