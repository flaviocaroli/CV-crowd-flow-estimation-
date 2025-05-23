import torch.nn as nn
import torchvision.models as models
from .unet_comp import DoubleConv, Up, CustomOutConv


class ResNetUNet(nn.Module):
    """
    ResNet50-based U-Net that matches UNet behavior exactly:
    - Full resolution output (H x W) like UNet, or optionally half resolution (H/2 x W/2)
    - Same layer structure as UNet but with ResNet encoder features
    - Variable depth (1-4, matching ResNet layers available)
    - Custom head support
    - No dropout parameter
    """
    def __init__(self, depth: int = 4, custom_head: bool = False, half_resolution: bool = False, **kwargs):
        super().__init__()
        assert 1 <= depth <= 4, "depth must be between 1 and 4 for ResNet layers"
        self.depth = depth
        self.half_resolution = half_resolution
        
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

        # Build ascending path - skip last decoder if half_resolution=True
        self.ups = nn.ModuleList()
        decoder_depth = depth - 1 if half_resolution else depth
        
        for i in reversed(range(decoder_depth)):
            in_ch = self.channels[i+1] + self.channels[i]  # decoder + skip
            out_ch = self.channels[i]
            self.ups.append(Up(in_ch, out_ch, **kwargs))
            
        # Output head - use appropriate channels based on resolution
        output_channels = self.channels[1] if half_resolution else self.channels[0]
        
        if custom_head:
            self.outc = CustomOutConv(output_channels, **kwargs)
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(output_channels, 1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x, return_intermediates=False):
        # Encoder (exactly like UNet structure)
        x_enc = [self.inc(x)]  # [B, 64, H, W]
        
        # Apply downsampling blocks
        for down in self.downs:
            x_enc.append(down(x_enc[-1]))
        
        # Bottleneck
        x_dec = self.bottleneck(x_enc[-1])

        # Decoder - skip connections start from second downsampling if half_resolution
        intermediates = []
        skip_start_idx = 1 if self.half_resolution else 0
        skip_connections = reversed(x_enc[skip_start_idx:-1])
        
        for up, skip in zip(self.ups, skip_connections):
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