import torch.nn as nn
import torchvision.models as models
from .unet_comp import DoubleConv, Down, Up, CustomOutConv


class VGGUNet(nn.Module):
    """
    VGG19-based U-Net that matches UNet behavior exactly:
    - Full resolution output (H x W) like UNet
    - Same layer structure as UNet but with VGG encoder features
    - Variable depth (1-5, matching VGG blocks)
    - Custom head support
    - No dropout parameter
    """
    def __init__(self, depth: int = 4, custom_head: bool = False, **kwargs):
        super().__init__()
        assert 1 <= depth <= 5, "depth must be between 1 and 5"
        self.depth = depth

        # Load pretrained VGG19 with batch-norm
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())
        
        # Extract VGG encoder blocks using same indices as VGG19BNBackbone
        all_vgg_blocks = [
            nn.Sequential(*feats[0:6]),    # 3->64 channels
            nn.Sequential(*feats[7:13]),   # 64->128 channels  
            nn.Sequential(*feats[14:23]),  # 128->256 channels
            nn.Sequential(*feats[24:33]),  # 256->512 channels
            nn.Sequential(*feats[34:43]),  # 512->512 channels
        ]
        
        # Use only first depth blocks
        self.vgg_blocks = all_vgg_blocks[:depth]
        
        # VGG19 channel progression (decoder outputs)
        all_decoder_channels = [64, 128, 256, 512, 512]
        self.encoder_in_channels = [3, 64, 128, 256, 512]
        self.decoder_channels = all_decoder_channels[:depth]
        
        # Build downsampling path using VGG blocks
        self.downs = nn.ModuleList()
        for i, vgg_block in enumerate(self.vgg_blocks):
            if i == 0:
                # First block: no pooling, direct VGG block
                self.downs.append(vgg_block)
            else:
                # Other blocks: pool then VGG block
                self.downs.append(VGGDown(vgg_block))
        
        # Bottleneck
        final_ch = self.decoder_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Build ascending path
        self.ups = nn.ModuleList()
        for i in reversed(range(depth-1)):
            in_ch = self.decoder_channels[i+1] + self.decoder_channels[i]  # decoder + skip
            out_ch = self.decoder_channels[i]
            self.ups.append(Up(in_ch, out_ch, **kwargs))
            
        # Output head - decoder outputs 64 channels (first VGG block output)
        if custom_head:
            self.outc = CustomOutConv(self.decoder_channels[0], **kwargs)
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(self.decoder_channels[0], 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, return_intermediates=False):
        # Encoder
        x_enc = []
        
        # First block: no pooling
        x = self.downs[0](x)  # VGG block 1: 3->64
        x_enc.append(x)
        
        # Remaining blocks: pool then VGG block
        for down in self.downs[1:]:
            x = down(x)
            x_enc.append(x)
        
        # Bottleneck
        x_dec = self.bottleneck(x_enc[-1])

        # Decoder
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


class VGGDown(nn.Module):
    """Downscaling block that uses VGG blocks"""
    def __init__(self, vgg_block):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.vgg_block = vgg_block

    def forward(self, x):
        x = self.maxpool(x)
        return self.vgg_block(x)


if __name__ == "__main__":
    import torch
    
    # Test the VGG U-Net
    print("Testing VGGUNet...")
    model = VGGUNet(depth=4, custom_head=False)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 1, 256, 256]
    
    # Test with intermediates
    intermediates = model(x, return_intermediates=True)
    print("Number of intermediate outputs:", len(intermediates))
    for i, inter in enumerate(intermediates):
        print(f"Intermediate {i} shape:", inter.shape)