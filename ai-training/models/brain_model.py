"""
3D CNN model architecture for brain scan analysis.
Designed for self-supervised learning on healthy brain patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Conv3DBlock(nn.Module):
    """3D Convolutional block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu'
    ):
        super(Conv3DBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block."""
    
    def __init__(self, channels: int):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = Conv3DBlock(channels, channels, use_bn=True)
        self.conv2 = Conv3DBlock(channels, channels, use_bn=True, activation='none')
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Residual connection
        out = self.activation(out)
        return out


class BrainScanModel(nn.Module):
    """
    3D CNN model for brain scan analysis.
    Architecture: Encoder-Decoder with skip connections for reconstruction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        base_channels: int = 32
    ):
        """
        Initialize model.
        
        Args:
            in_channels: Number of input channels (1 for T1)
            feature_dim: Dimension of encoded features
            base_channels: Base number of channels in first layer
        """
        super(BrainScanModel, self).__init__()
        
        self.feature_dim = feature_dim
        self.base_channels = base_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            # Initial conv
            Conv3DBlock(in_channels, base_channels, stride=2),  # 64x64x64
            Conv3DBlock(base_channels, base_channels),
            
            # Downsample 1
            Conv3DBlock(base_channels, base_channels * 2, stride=2),  # 32x32x32
            ResidualBlock3D(base_channels * 2),
            
            # Downsample 2
            Conv3DBlock(base_channels * 2, base_channels * 4, stride=2),  # 16x16x16
            ResidualBlock3D(base_channels * 4),
            
            # Downsample 3
            Conv3DBlock(base_channels * 4, base_channels * 8, stride=2),  # 8x8x8
            ResidualBlock3D(base_channels * 8),
            
            # Global average pooling
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            
            # Feature projection
            nn.Linear(base_channels * 8, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, base_channels * 8 * 8 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (base_channels * 8, 8, 8, 8)),
            
            # Upsample 1
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2),
            ResidualBlock3D(base_channels * 4),  # 16x16x16
            
            # Upsample 2
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            ResidualBlock3D(base_channels * 2),  # 32x32x32
            
            # Upsample 3
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            ResidualBlock3D(base_channels),  # 64x64x64
            
            # Upsample 4
            nn.ConvTranspose3d(base_channels, base_channels, kernel_size=2, stride=2),
            Conv3DBlock(base_channels, base_channels),  # 128x128x128
            
            # Final output
            nn.Conv3d(base_channels, in_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Feature projection head (for downstream tasks)
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input volume to feature representation.
        
        Args:
            x: Input volume tensor (B, C, D, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        return self.encoder(x)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to volume.
        
        Args:
            features: Feature tensor (B, feature_dim)
            
        Returns:
            Reconstructed volume (B, C, D, H, W)
        """
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, decode, and extract features.
        
        Args:
            x: Input volume tensor (B, C, D, H, W)
            
        Returns:
            Tuple of (reconstructed volume, encoded features, projected features)
        """
        # Encode
        encoded = self.encode(x)
        
        # Decode for reconstruction
        reconstructed = self.decode(encoded)
        
        # Extract features for downstream tasks
        features = self.feature_head(encoded)
        
        return reconstructed, encoded, features
    
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representation without decoding (for inference).
        
        Args:
            x: Input volume tensor
            
        Returns:
            Feature representation
        """
        with torch.no_grad():
            encoded = self.encode(x)
            features = self.feature_head(encoded)
        return features


def create_model(
    in_channels: int = 1,
    feature_dim: int = 256,
    base_channels: int = 32
) -> BrainScanModel:
    """
    Factory function to create model.
    
    Args:
        in_channels: Input channels
        feature_dim: Feature dimension
        base_channels: Base channels
        
    Returns:
        Initialized model
    """
    model = BrainScanModel(
        in_channels=in_channels,
        feature_dim=feature_dim,
        base_channels=base_channels
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model

