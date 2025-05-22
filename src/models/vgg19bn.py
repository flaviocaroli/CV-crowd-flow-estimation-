import torch.nn as nn
import torchvision.models as models
from .unet_comp import Up, CustomOutConv
import torchsummary
class VGG19BNBackbone(nn.Module):
    """
    U-Net for density regression using VGG19_bn encoder.
    Outputs [B,1,H/2,W/2] given [B,3,H,W].
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())
        # Encoder blocks + pools
        self.enc1, self.pool1 = nn.Sequential(*feats[0:6]), feats[6]
        self.enc2, self.pool2 = nn.Sequential(*feats[7:13]), feats[13]
        self.enc3, self.pool3 = nn.Sequential(*feats[14:23]), feats[23]
        self.enc4, self.pool4 = nn.Sequential(*feats[24:33]), feats[33]
        self.enc5, self.pool5 = nn.Sequential(*feats[34:43]), feats[43]
        
        # Decoder (same for both modes)
        # up1: 512 + 512 → 512
        # up2: 512 + 512 → 256
        # up3: 256 + 256 → 128
        # up4: 128 + 128 → 64   ← note skip from enc2 (128-ch) so output is half-res
        self.up1   = Up(512+512, 512)
        self.up2   = Up(512+512, 256)
        self.up3   = Up(256+256, 128)
        self.up4   = Up(128+128,  64)
        self.outc  = nn.Conv2d(64, 1, kernel_size=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, return_intermediates: bool = False):
        
        x1 = self.enc1(x)
        x2 = self.pool1(x1); x2 = self.enc2(x2)
        x3 = self.pool2(x2); x3 = self.enc3(x3)
        x4 = self.pool3(x3); x4 = self.enc4(x4)
        x5 = self.pool4(x4); x5 = self.enc5(x5)
        x6 = self.pool5(x5)

        d1 = self.up1(x6, x5)
        d2 = self.up2(d1, x4)
        d3 = self.up3(d2, x3)
        d4 = self.up4(d3, x2)   # ← skip from x2 for half-res output
        out = self.relu(self.outc(d4))

        if return_intermediates:
            return [d1, d2, d3, d4, out]
        return out

class VGGUNet(nn.Module):
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



if __name__ == "__main__":
    from torchsummary import summary
    import torch


    # Test the model at various depths
    for depth in range(1, 6):
        print(f"\nTesting VGGUNet with depth={depth}")
        model = VGGUNet(depth=depth, custom_head=False)
        print(model)
        x = torch.randn(1, 3, 256, 256)
        output = model(x, return_intermediates=True)
        assert output[-1].shape[3] == 128, f"Output shape mismatch for depth {depth}"
        print(f"Output shape: {output[-1].shape}")
        for i, inter in enumerate(output[:-1]):
            print(f"Intermediate {i} shape: {inter.shape}")

    # Assert VGGUNet(depth=5) matches VGG19BNBackbone structure
    vgg_old = VGG19BNBackbone()
    vgg_new = VGGUNet(depth=4, custom_head=False)
    # Compare string representations (structure only, not weights)
    print("\nComparing VGG19BNBackbone and VGGUNet(depth=5) structures...")
    print("VGG19BNBackbone structure:")
    print(vgg_old)
    print("VGGUNet(depth=5) structure:")
    print(vgg_new)
    assert str(vgg_old) == str(vgg_new), "VGGUNet(depth=5) does not match VGG19BNBackbone structure"

    print("\nAssertion passed: VGGUNet(depth=5) matches VGG19BNBackbone structure.")