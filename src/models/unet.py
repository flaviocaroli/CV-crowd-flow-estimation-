import torch.nn as nn
from .unet_comp import DoubleConv, Down, Up, CustomOutConv

class UNetSkip1(nn.Module):
    """
    Generic U-Net with variable depth D.
    - in_channels: # input channels (e.g. 3 for RGB)
    - num_filters: F, number of features at the first level
    - depth: number of down/up steps (positive integer)
    
    forward(x, return_intermediates=False):
      - x: [B, in_channels, H, W]
      - returns [B,1,H,W], or if return_intermediates,
        a list of decoder outputs [d1, d2, ..., dD, final_out].
    """
    def __init__(self, in_channels=3, num_filters=32, depth=4, **kwargs):
        super().__init__()
        assert depth >= 1, "Depth must be >= 1"
        assert num_filters > 0, "Base channels must be > 0"
        # check that if you divide the input size by 2^depth, you get a positive integer
        custom_head = kwargs.get("custom_head", False)
        self.depth = depth
        F = num_filters
        
        self.inc = nn.Sequential(
            DoubleConv(in_channels, F, **kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # build descending path
        self.downs = nn.ModuleList()
        for i in range(depth):
            in_ch = F * (2**i)
            out_ch = F * (2**(i+1))
            self.downs.append(Down(in_ch, out_ch, **kwargs))
        
        # build ascending path
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            # at this up-step, we combine:
            #   - feature from the level below: F*2^(i+1)
            #   - skip feature from encoder: F*2^i
            in_ch = F*(2**(i+1)) + F*(2**i)
            out_ch = F*(2**i)
            self.ups.append(Up(in_ch, out_ch, **kwargs))
        if custom_head:
            self.outc = CustomOutConv(F, **kwargs)
        else:
            # final 1x1 conv to map F channels -> 1
            self.outc = nn.Sequential(
                nn.Conv2d(F, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, return_intermediates=False):
        # Encoder
        x_enc = [self.inc(x)]  # x_enc[0] has shape [B, F,   H,   W]
        for down in self.downs:
            x_enc.append(down(x_enc[-1]))
        # x_enc[k] has shape [B, F*2^k, H/2^k, W/2^k],  k=0..depth
        
        # Decoder
        intermediates = []
        x_dec = x_enc[-1]  # bottom feature
        for up, skip in zip(self.ups, reversed(x_enc[:-1])):
            x_dec = up(x_dec, skip)
            if return_intermediates:
                intermediates.append(x_dec)
        
        out = self.outc(x_dec)
        if return_intermediates:
            intermediates.append(out)
            return intermediates
        
        return out


class UNet(nn.Module):
    """
    Generic U-Net with variable depth D.
    - in_channels: # input channels (e.g. 3 for RGB)
    - num_filters: F, number of features at the first level
    - depth: number of down/up steps (positive integer)
    
    forward(x, return_intermediates=False):
      - x: [B, in_channels, H, W]
      - returns [B,1,H,W], or if return_intermediates,
        a list of decoder outputs [d1, d2, ..., dD, final_out].
    """
    def __init__(self, in_channels=3, num_filters=32, depth=4, **kwargs):
        super().__init__()
        assert depth >= 1, "Depth must be >= 1"
        assert num_filters > 0, "Base channels must be > 0"
        custom_head = kwargs.get("custom_head", False)
        
        self.depth_dilation: int = kwargs.get("depth_dilation", 1)


        self.depth = depth
        F = num_filters
        
        # Initial convolution WITHOUT pooling
        self.inc = DoubleConv(in_channels, F, **kwargs)
        
        # Build descending path
        self.downs = nn.ModuleList()
        for i in range(depth):
            in_ch = F * (2**i)
            out_ch = F * (2**(i+1))
            if i > self.depth_dilation:
                updated_kwargs = kwargs.copy()
                updated_kwargs.pop("dilation_l1")
                updated_kwargs.pop("dilation_l2")
                self.downs.append(Down(in_ch, out_ch, **updated_kwargs))
        
        # Bottleneck
        self.bottleneck = DoubleConv(F*(2**depth), F*(2**depth), **kwargs)

        # Build ascending path
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            # at this up-step, we combine:
            #   - feature from the level below: F*2^(i+1)
            #   - skip feature from encoder: F*2^i
            in_ch = F*(2**(i+1)) + F*(2**i)
            out_ch = F*(2**i)
            self.ups.append(Up(in_ch, out_ch, **kwargs))
            
        if custom_head:
            self.outc = CustomOutConv(F, **kwargs)
        else:
            # Final 1x1 conv to map F channels -> 1
            self.outc = nn.Sequential(
                nn.Conv2d(F, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, return_intermediates=False):
        # Encoder
        x_enc = [self.inc(x)]  # x_enc[0] has shape [B, F, H, W]
        
        # Apply downsampling blocks
        for down in self.downs:
            x_enc.append(down(x_enc[-1]))
        # Now x_enc has depth+1 elements:
        # x_enc[0]: [B, F, H, W]
        # x_enc[1]: [B, F*2, H/2, W/2]
        # x_enc[2]: [B, F*4, H/4, W/4]
        # ...
        # x_enc[depth]: [B, F*2^depth, H/2^depth, W/2^depth]
        
        # Bottleneck
        x_dec = self.bottleneck(x_enc[-1])

        # Decoder - pair each up block with the corresponding encoder feature
        intermediates = []
        for i, (up, skip) in enumerate(zip(self.ups, reversed(x_enc[:-1]))):
            x_dec = up(x_dec, skip)
            if return_intermediates:
                intermediates.append(x_dec)
        
        out = self.outc(x_dec)
        if return_intermediates:
            intermediates.append(out)
            return intermediates
        
        return out