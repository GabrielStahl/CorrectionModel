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

class CorrectionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, dropout=config.dropout):
        super(CorrectionUNet, self).__init__()
        self.dropout = dropout

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 16, 2)
        self.enc2 = self.conv_block(16, 32, 2)
        self.enc3 = self.conv_block(32, 64, 2)
        self.enc4 = self.conv_block(64, 128, 2)
        self.enc5 = self.conv_block(128, 256, 1)
        self.enc6 = self.conv_block(256, 256, 1)
        self.enc7 = self.conv_block(256, 256, 1)

        # Decoder (upsampling)
        self.dec1 = self.conv_block(256 + 256, 256, 2, dropout=self.dropout)
        self.dec2 = self.conv_block(256 + 128, 128, 2, dropout=self.dropout)
        self.dec3 = self.conv_block(128 + 64, 64, 2)
        self.dec4 = self.conv_block(64 + 32, 32, 2)
        self.dec5 = nn.Conv3d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, num_conv, dropout=None):
        layers = []
        for i in range(num_conv):
            if i == 0:
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None:
                layers.append(nn.Dropout3d(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        enc5 = self.enc5(self.maxpool(enc4))
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)

        # Decoder
        dec1 = self.dec1(torch.cat([self.upsample(enc7, enc6.size()), enc6], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec1, enc4.size()), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec2, enc3.size()), enc3], dim=1))
        dec4 = self.dec4(torch.cat([self.upsample(dec3, enc2.size()), enc2], dim=1))
        output = self.dec5(self.upsample(dec4, x.size()))

        return output

    def maxpool(self, x):
        return nn.functional.max_pool3d(x, kernel_size=2, stride=1, padding=1)

    def upsample(self, x, size):
        return F.interpolate(x, size=size[2:], mode='trilinear', align_corners=True)