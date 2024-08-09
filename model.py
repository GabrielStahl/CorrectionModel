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
    def __init__(self, in_channels=config.in_channels, out_channels=config.out_channels, dropout=None): # 4 classes: tumour core, enhancing tumor, edema, background
        super(CorrectionUNet, self).__init__()

        self.dropout = dropout

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block_DOWN(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block_DOWN(128, 256)
        self.enc5 = self.conv_block(256, 320)
        self.enc6 = self.conv_block_DOWN(320, 320)
        self.enc7 = self.conv_block(320, 320)

        # Decoder (upsampling)
        self.dec7 = self.conv_block(320, 320, dropout=self.dropout)
        self.dec6 = self.conv_block_UP(320, 320, dropout=self.dropout)
        self.dec5 = self.conv_block(320, 256)
        self.dec4 = self.conv_block_UP(256 + 256, 128)
        self.dec3 = self.conv_block(128 + 128, 64)
        self.dec2 = self.conv_block_UP(64 + 64, 32)
        self.dec1 = nn.Conv3d(32 + 32, out_channels, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def conv_block_DOWN(self, in_channels, out_channels, dropout=None):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        ]
        
        if dropout is not None:
            layers.append(nn.Dropout3d(p=dropout))
        
        return nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels, dropout=None):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ]
        
        if dropout is not None:
            layers.append(nn.Dropout3d(p=dropout))
        
        return nn.Sequential(*layers)
    
    def conv_block_UP(self, in_channels, out_channels, dropout=None):
        layers = [
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ]
        
        if dropout is not None:
            layers.append(nn.Dropout3d(p=dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)

        # Decoder
        dec7 = self.dec7(enc7)
        dec6 = self.dec6(dec7)
        dec5 = self.dec5(dec6) 

        # crop and concatenate
        enc4 = self.crop(enc4, dec5.size()[2:])
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        enc3 = self.crop(enc3, dec4.size()[2:])
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        enc2 = self.crop(enc2, dec3.size()[2:])
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        enc1 = self.crop(enc1, dec2.size()[2:])
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        return dec1 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        target_depth, target_height, target_width = target_size

        crop_depth = (layer_depth - target_depth) // 2
        crop_height = (layer_height - target_height) // 2
        crop_width = (layer_width - target_width) // 2

        return layer[:, :, crop_depth:crop_depth + target_depth, crop_height:crop_height + target_height, crop_width:crop_width + target_width]