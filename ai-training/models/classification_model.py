"""
2D CNN model architecture for brain tumor image classification.
Designed for multi-class classification of brain tumors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torchvision.models as models


class Conv2DBlock(nn.Module):
    """2D Convolutional block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super(Conv2DBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock2D(nn.Module):
    """2D Residual block."""
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super(ResidualBlock2D, self).__init__()
        
        self.conv1 = Conv2DBlock(channels, channels, use_bn=True, dropout=0.0)
        self.conv2 = Conv2DBlock(channels, channels, use_bn=True, activation='none', dropout=dropout)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Residual connection
        out = self.activation(out)
        return out


class BrainTumorClassificationModel(nn.Module):
    """
    2D CNN model for brain tumor image classification.
    Architecture: Custom CNN with residual blocks for 4-class classification.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 64,
        use_pretrained: bool = False,
        model_type: str = 'custom'
    ):
        """
        Initialize model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            base_channels: Base number of channels in first conv layer
            use_pretrained: Whether to use pretrained ResNet (if model_type='resnet')
            model_type: 'custom' or 'resnet'
        """
        super(BrainTumorClassificationModel, self).__init__()
        
        self.num_classes = num_classes
        self.model_type = model_type
        
        if model_type == 'resnet':
            # Use ResNet as backbone
            try:
                # Try new API (torchvision >= 0.13)
                if use_pretrained:
                    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                else:
                    resnet = models.resnet50(weights=None)
            except (AttributeError, TypeError):
                # Fallback to old API (torchvision < 0.13)
                if use_pretrained:
                    resnet = models.resnet50(pretrained=True)
                else:
                    resnet = models.resnet50(pretrained=False)
            
            # Replace first layer if needed (ResNet expects 3 channels by default)
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Remove the final fc layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            # Custom CNN architecture
            self.features = nn.Sequential(
                # Initial conv
                Conv2DBlock(in_channels, base_channels, kernel_size=7, stride=2, padding=3),  # 112x112
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56
                
                # Block 1
                Conv2DBlock(base_channels, base_channels),
                ResidualBlock2D(base_channels, dropout=0.1),
                
                # Downsample 1
                Conv2DBlock(base_channels, base_channels * 2, stride=2),  # 28x28
                ResidualBlock2D(base_channels * 2, dropout=0.1),
                
                # Downsample 2
                Conv2DBlock(base_channels * 2, base_channels * 4, stride=2),  # 14x14
                ResidualBlock2D(base_channels * 4, dropout=0.2),
                
                # Downsample 3
                Conv2DBlock(base_channels * 4, base_channels * 8, stride=2),  # 7x7
                ResidualBlock2D(base_channels * 8, dropout=0.2),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(base_channels * 8, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Class logits (B, num_classes)
        """
        if self.model_type == 'resnet':
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        else:
            features = self.features(x)
        
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input image tensor
            
        Returns:
            Class probabilities (B, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input image tensor
            
        Returns:
            Predicted class indices (B,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_classification_model(
    in_channels: int = 3,
    num_classes: int = 4,
    base_channels: int = 64,
    use_pretrained: bool = False,
    model_type: str = 'custom'
) -> BrainTumorClassificationModel:
    """
    Factory function to create classification model.
    
    Args:
        in_channels: Input channels
        num_classes: Number of classes
        base_channels: Base channels (for custom model)
        use_pretrained: Whether to use pretrained weights (for ResNet)
        model_type: 'custom' or 'resnet'
        
    Returns:
        Initialized model
    """
    model = BrainTumorClassificationModel(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        use_pretrained=use_pretrained,
        model_type=model_type
    )
    
    # Initialize weights (only for custom model or non-pretrained ResNet)
    if model_type == 'custom' or not use_pretrained:
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
    
    return model

