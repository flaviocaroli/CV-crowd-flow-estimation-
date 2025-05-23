import torch.nn as nn
import torchvision.models as models
from .unet_comp import DoubleConv, Up, CustomOutConv

class VGGSkip1(nn.Module):
    """
    VGG-based U-Net with halved-resolution output (H/2 x W/2), and variable depth:
    - depth: number of encoder+pool blocks to use (1–5)
    - uses the first `depth` conv-blocks from VGG19_bn, each followed by its MaxPool
    - performs (depth-1) ups to return to H/2, matching VGG19BNBackbone structure
    - custom_head: if True, use CustomOutConv; else a 1×1 conv + ReLU
    """
    def __init__(self, depth: int = 5, custom_head: bool = False, **kwargs):
        super().__init__()
        assert 1 <= depth <= 5, "depth must be between 1 and 5"
        self.depth = depth
        self.custom_head = custom_head

        # Load pretrained VGG19 with batch-norm
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        feats = list(vgg.features)

        # Split into convolutional blocks and pool layers
        conv_blocks_all, pool_blocks, current = [], [], []
        for layer in feats:
            if isinstance(layer, nn.MaxPool2d):
                conv_blocks_all.append(nn.Sequential(*current))
                pool_blocks.append(layer)
                current = []
            else:
                current.append(layer)

        # Keep only the first `depth` blocks for the encoder
        self.encoder_conv_blocks = nn.ModuleList(conv_blocks_all[:depth])
        self.encoder_pool_blocks = nn.ModuleList(pool_blocks[:depth])

        # Record output channels of each encoder block
        self.channels = [block[0].out_channels for block in self.encoder_conv_blocks]

        # Build decoder: exactly (depth-1) Ups to restore to H/2, with correct skip channels
        self.ups = nn.ModuleList()
        for j in range(depth - 1, 0, -1):
            ch = self.channels[j]
            out_ch = self.channels[j - 1]
            # merge bottom or previous up output (ch) with skip feature (ch)
            self.ups.append(Up(ch * 2, out_ch))

        # Output head
        if custom_head:
            self.outc = CustomOutConv(self.channels[0], **kwargs)
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(self.channels[0], 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, return_intermediates: bool = False):
        # Encoder: collect skip features
        x_enc = []
        for conv, pool in zip(self.encoder_conv_blocks, self.encoder_pool_blocks):
            x = conv(x)
            x_enc.append(x)
            x = pool(x)

        # Bottom feature at H/(2^depth)
        x_dec = x
        intermediates = []

        # Decoder: (depth-1) ups, skipping conv2..conv_depth
        skip_feats = x_enc[1:]
        for up, skip in zip(self.ups, reversed(skip_feats)):
            x_dec = up(x_dec, skip)
            if return_intermediates:
                intermediates.append(x_dec)

        out = self.outc(x_dec)
        if return_intermediates:
            intermediates.append(out)
            return intermediates
        return out



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
        self.bottleneck = DoubleConv(final_ch, final_ch, **kwargs)

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