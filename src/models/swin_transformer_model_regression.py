"""
Swin Transformer model with regression for SNR prediction.
Updated to use regression instead of classification for SNR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SwinTransformerRegressionModel(nn.Module):
    """
    Swin Transformer model for joint modulation classification and SNR regression.
    
    Key changes:
    1. SNR prediction is now regression (single output)
    2. Dropout applied consistently throughout the model
    3. Simplified architecture without backward compatibility
    """
    
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        num_classes=17,  # Modulation classes
        pretrained=True,
        input_channels=1,
        dropout_prob=0.3,
        use_dilated_preprocessing=True,
    ):
        """
        Initialize the Swin Transformer model for regression.
        
        Args:
            model_name (str): Name of the Swin model variant.
            num_classes (int): Number of modulation classes.
            pretrained (bool): Whether to use pretrained weights.
            input_channels (int): Number of input channels.
            dropout_prob (float): Dropout probability used throughout the model.
            use_dilated_preprocessing (bool): Whether to use dilated CNN preprocessing.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_dilated_preprocessing = use_dilated_preprocessing
        self.dropout_prob = dropout_prob
        
        # Create the base Swin model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Dilated CNN preprocessing (if enabled)
        if use_dilated_preprocessing:
            self.dilated_preprocessing = nn.Sequential(
                # Layer 1: Local features (RF: 3x3)
                nn.Conv2d(input_channels, 32, kernel_size=3, dilation=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_prob),
                
                # Layer 2: Neighborhood patterns (RF: 7x7)
                nn.Conv2d(32, 64, kernel_size=3, dilation=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_prob),
                
                # Layer 3: Inter-cluster patterns (RF: 15x15)
                nn.Conv2d(64, 96, kernel_size=3, dilation=4, padding=4),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_prob),
                
                # Layer 4: Global constellation spread (RF: 31x31)
                nn.Conv2d(96, 96, kernel_size=3, dilation=8, padding=8),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_prob),
                
                # Compress to 3 channels for Swin input
                nn.Conv2d(96, 3, kernel_size=1, padding=0)
            )
            swin_input_channels = 3
        else:
            swin_input_channels = input_channels
        
        # Modify Swin input layer if needed
        if swin_input_channels != 3:
            first_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                swin_input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
        
        # Get feature dimension
        self.feature_dim = self.model.head.in_features
        
        # Remove the existing classifier head
        self.model.head = nn.Identity()
        
        # Modulation classification head
        self.modulation_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        # SNR regression head (single output)
        self.snr_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1)  # Single output for regression
        )
    
    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            dict: Dictionary with 'modulation' logits and 'snr' regression value
        """
        # Apply dilated preprocessing if enabled
        if self.use_dilated_preprocessing:
            x = self.dilated_preprocessing(x)
        
        # Extract features using Swin Transformer
        features = self.model(x)
        
        # Apply dropout after Swin backbone for regularization
        features = F.dropout(features, p=self.dropout_prob, training=self.training)
        
        # Modulation classification
        modulation_logits = self.modulation_head(features)
        
        # SNR regression
        snr_value = self.snr_head(features)
        
        return {
            'modulation': modulation_logits,
            'snr': snr_value
        }


def create_swin_regression_model(
    model_variant="swin_tiny",
    num_classes=17,
    pretrained=True,
    input_channels=1,
    dropout_prob=0.3,
    use_dilated_preprocessing=True,
):
    """
    Factory function to create Swin Transformer regression models.
    
    Args:
        model_variant (str): Which Swin variant to use ('swin_tiny', 'swin_small', 'swin_base')
        num_classes (int): Number of modulation classes
        pretrained (bool): Whether to use pretrained weights
        input_channels (int): Number of input channels
        dropout_prob (float): Dropout probability used throughout the model
        use_dilated_preprocessing (bool): Whether to use dilated CNN preprocessing
        
    Returns:
        SwinTransformerRegressionModel: The initialized model
    """
    model_map = {
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
    }
    
    if model_variant not in model_map:
        raise ValueError(f"Model variant {model_variant} not supported. Choose from {list(model_map.keys())}")
    
    return SwinTransformerRegressionModel(
        model_name=model_map[model_variant],
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels,
        dropout_prob=dropout_prob,
        use_dilated_preprocessing=use_dilated_preprocessing,
    )