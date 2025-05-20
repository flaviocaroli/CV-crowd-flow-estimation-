import torch.nn as nn
from .unet_comp import DoubleConv, Down, Up

class UNet(nn.Module):
    """
    Generic U-Net with variable depth D.
    - in_channels: # input channels (e.g. 3 for RGB)
    - base_channels: F, number of features at the first level
    - depth: number of down/up steps (positive integer)
    
    forward(x, return_intermediates=False):
      - x: [B, in_channels, H, W]
      - returns [B,1,H,W], or if return_intermediates,
        a list of decoder outputs [d1, d2, ..., dD, final_out].
    """
    def __init__(self, in_channels=3, base_channels=32, depth=4):
        super().__init__()
        assert depth >= 1, "Depth must be >= 1"
        self.depth = depth
        F = base_channels
        
        self.inc = nn.Sequential(
            DoubleConv(in_channels, F),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # build descending path
        self.downs = nn.ModuleList()
        for i in range(depth):
            in_ch = F * (2**i)
            out_ch = F * (2**(i+1))
            self.downs.append(Down(in_ch, out_ch))
        
        # build ascending path
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            # at this up-step, we combine:
            #   - feature from the level below: F*2^(i+1)
            #   - skip feature from encoder: F*2^i
            in_ch = F*(2**(i+1)) + F*(2**i)
            out_ch = F*(2**i)
            self.ups.append(Up(in_ch, out_ch))
        
        # final 1x1 conv to map F channels -> 1
        self.outc = nn.Conv2d(F, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

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
        out = self.relu(out)
        if return_intermediates:
            intermediates.append(out)
            return intermediates
        
        return out