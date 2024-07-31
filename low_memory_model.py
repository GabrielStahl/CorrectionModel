import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import os
import warnings

# Set the environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Suppress warnings
warnings.filterwarnings("ignore")

class UltraLightCorrectionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(UltraLightCorrectionUNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        
        # Decoder (upsampling)
        self.dec3 = self.conv_block(128 + 64, 64)
        self.dec2 = self.conv_block(64 + 32, 32)
        self.dec1 = self.conv_block(32 + 16, 16)
        
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        
        # Decoder
        dec3 = self.dec3(torch.cat([self.upsample(enc4, enc3.size()), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3, enc2.size()), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2, enc1.size()), enc1], dim=1))
        
        output = self.final_conv(dec1)
        return output

    def maxpool(self, x):
        return nn.functional.max_pool3d(x, kernel_size=2, stride=2)

    def upsample(self, x, size):
        return F.interpolate(x, size=size[2:], mode='nearest')