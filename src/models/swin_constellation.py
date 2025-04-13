"""
Implementation of Swin Transformer for Constellation Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b
from torchvision.models.swin_transformer import SwinTransformer


class SwinConstellation(nn.Module):
    """
    Swin Transformer for Constellation Recognition
    Uses the standard Swin Transformer architecture with modifications
    for grayscale input and custom classification heads
    """
    def __init__(self, num_classes=11, num_snr_classes=None, dropout_prob=0.3):
        super().__init__()
        
        # Load the standard Swin-B model
        base_model = swin_b(weights=None)
        
        # Get the config from the base model to rebuild first layer
        original_model = base_model.features[0][0]
        embed_dim = original_model.out_channels
        patch_size = original_model.kernel_size
        stride = original_model.stride
        padding = original_model.padding
        
        # Create new patch embedding layer for grayscale (1 channel instead of 3)
        # Input: (B, 1, H, W) -> Output: (B, embed_dim, H/patch_size, W/patch_size)
        new_patch_embedding = nn.Conv2d(
            in_channels=1,  # Grayscale input
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        
        # Replace the first conv layer but keep everything else
        base_model.features[0][0] = new_patch_embedding
        
        # Store the modified model as our backbone
        self.backbone = base_model
        
        # The Swin-B model outputs 1000 features (ImageNet classes)
        swin_output_dim = 1000
        
        # Task-specific heads
        # Input: (B, 1000) -> Output: (B, num_classes)
        self.modulation_head = nn.Sequential(
            nn.Linear(swin_output_dim, 512),  # 1000 is the output dim of Swin-B
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        if num_snr_classes is None:
            num_snr_classes = num_classes
        
        # Input: (B, 1000) -> Output: (B, num_snr_classes)    
        self.snr_head = nn.Sequential(
            nn.Linear(swin_output_dim, 512),  # 1000 is the output dim of Swin-B
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_snr_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            tuple: (modulation_output, snr_output)
        """
        # Input shape: (B, 1, H, W)
        B = x.shape[0]
        
        # Using the Swin backbone directly
        # Input: (B, 1, H, W) -> Output: (B, 1000)
        features = self.backbone(x)
        
        # Task-specific heads
        # Input: (B, 1000) -> Output: (B, num_classes)
        modulation_output = self.modulation_head(features)
        
        # Input: (B, 1000) -> Output: (B, num_snr_classes)
        snr_output = self.snr_head(features)
        
        return modulation_output, snr_output 