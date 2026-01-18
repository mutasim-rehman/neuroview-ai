"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Provides visual localization of where the model is "looking" for tumor detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional, Dict, Any


class GradCAM:
    """
    Grad-CAM implementation for visual tumor localization.
    
    Generates heatmaps showing which regions of the input image
    contributed most to the model's prediction.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The convolutional layer to compute CAM from
                         (typically the last conv layer before pooling)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
            - heatmap: numpy array (H, W) with values 0-1
            - predicted_class: int class index
            - confidence: float probability
        """
        self.model.eval()
        
        # Enable gradients for input
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probabilities[0, target_class].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class, confidence
    
    def generate_heatmap_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: str = 'jet'
    ) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
        """
        Generate heatmap overlay on original image.
        
        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original PIL Image (for overlay)
            target_class: Target class index (None = use prediction)
            alpha: Overlay transparency (0-1)
            colormap: Matplotlib colormap name
            
        Returns:
            Tuple of (overlay_image, raw_heatmap, metadata)
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Generate CAM
        cam, pred_class, confidence = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to match original image size
        cam_resized = np.array(Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).resize(original_image.size, Image.BILINEAR)) / 255.0
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Convert original image to RGB if needed
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        original_array = np.array(original_image)
        
        # Create overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * original_array).astype(np.uint8)
        overlay_image = Image.fromarray(overlay)
        
        # Find tumor location (center of mass of high activation)
        location_info = self._extract_location_from_cam(cam_resized)
        
        metadata = {
            'predicted_class': pred_class,
            'confidence': confidence,
            'location_info': location_info
        }
        
        return overlay_image, cam_resized, metadata
    
    def _extract_location_from_cam(self, cam: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extract location information from CAM heatmap.
        
        Args:
            cam: CAM heatmap (H, W) with values 0-1
            threshold: Threshold for considering high activation
            
        Returns:
            Dictionary with location information
        """
        h, w = cam.shape
        
        # Threshold the CAM to get high activation regions
        high_activation = cam > threshold
        
        if not high_activation.any():
            # Lower threshold if nothing found
            threshold = 0.3
            high_activation = cam > threshold
        
        if not high_activation.any():
            return {
                'detected': False,
                'message': 'No significant activation detected'
            }
        
        # Find center of mass of high activation
        y_coords, x_coords = np.where(high_activation)
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
        
        # Calculate relative position (0-1)
        rel_x = center_x / w
        rel_y = center_y / h
        
        # Determine quadrant/region
        horizontal = 'left' if rel_x < 0.33 else ('center' if rel_x < 0.66 else 'right')
        vertical = 'superior' if rel_y < 0.33 else ('middle' if rel_y < 0.66 else 'inferior')
        
        # Calculate activation area percentage
        activation_area = np.sum(high_activation) / (h * w) * 100
        
        # Find bounding box of high activation
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        
        return {
            'detected': True,
            'center': {
                'x': float(rel_x),
                'y': float(rel_y),
                'x_pixel': int(center_x),
                'y_pixel': int(center_y)
            },
            'region': f'{vertical}-{horizontal}',
            'quadrant': f'{vertical} {horizontal} region',
            'activation_area_percent': round(activation_area, 2),
            'bounding_box': {
                'x_min': int(min_x),
                'y_min': int(min_y),
                'x_max': int(max_x),
                'y_max': int(max_y),
                'width': int(max_x - min_x),
                'height': int(max_y - min_y)
            },
            'max_activation': float(cam.max()),
            'mean_activation': float(cam[high_activation].mean())
        }


def get_target_layer(model, model_type: str = 'custom'):
    """
    Get the target convolutional layer for Grad-CAM.
    
    Args:
        model: The model
        model_type: 'custom' or 'resnet'
        
    Returns:
        Target layer for Grad-CAM
    """
    if model_type == 'resnet':
        # For ResNet, use the last conv layer in backbone
        # backbone is nn.Sequential of ResNet layers
        return model.backbone[-2][-1].conv2  # Last conv in last ResNet block
    else:
        # For custom model, get the last conv layer before pooling
        # features is nn.Sequential, find last Conv2DBlock
        for i in range(len(model.features) - 1, -1, -1):
            layer = model.features[i]
            if hasattr(layer, 'conv'):  # Conv2DBlock or ResidualBlock2D
                return layer.conv if hasattr(layer, 'conv') else layer.conv2
        # Fallback: return features[-3] which should be a conv layer
        return model.features[-4]  # Before AdaptiveAvgPool2d and Flatten


def generate_gradcam_visualization(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    device: torch.device,
    model_type: str = 'custom',
    target_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate Grad-CAM visualization for tumor localization.
    
    Args:
        model: Trained classification model
        input_tensor: Preprocessed input tensor (1, C, H, W)
        original_image: Original PIL Image
        device: Torch device
        model_type: 'custom' or 'resnet'
        target_class: Target class (None = use prediction)
        
    Returns:
        Dictionary with:
        - heatmap_base64: Base64 encoded heatmap overlay image
        - raw_heatmap_base64: Base64 encoded raw heatmap
        - location_info: Extracted location information
    """
    try:
        # Get target layer
        target_layer = get_target_layer(model, model_type)
        
        # Create GradCAM instance
        gradcam = GradCAM(model, target_layer)
        
        # Move input to device
        input_tensor = input_tensor.to(device)
        
        # Generate heatmap overlay
        overlay_image, raw_heatmap, metadata = gradcam.generate_heatmap_overlay(
            input_tensor,
            original_image,
            target_class=target_class,
            alpha=0.5
        )
        
        # Convert overlay to base64
        overlay_buffer = io.BytesIO()
        overlay_image.save(overlay_buffer, format='PNG')
        overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
        
        # Convert raw heatmap to base64 (grayscale)
        raw_heatmap_img = Image.fromarray((raw_heatmap * 255).astype(np.uint8), mode='L')
        raw_buffer = io.BytesIO()
        raw_heatmap_img.save(raw_buffer, format='PNG')
        raw_base64 = base64.b64encode(raw_buffer.getvalue()).decode('utf-8')
        
        return {
            'heatmap_overlay_base64': overlay_base64,
            'raw_heatmap_base64': raw_base64,
            'localization': metadata['location_info'],
            'success': True
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
