# src/models/vision_transformer_model.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ConstellationVisionTransformer(nn.Module):
    """
    A Vision Transformer (ViT) model for constellation classification.
    Uses a pre-trained ViT model as a base and adds task-specific heads.
    """

    def __init__(self, num_classes=11, snr_classes=26, input_channels=1, dropout_prob=0.3):
        """
        Initialize the ConstellationVisionTransformer model.

        Args:
            num_classes (int): Number of modulation classes.
            snr_classes (int): Not used anymore - kept for compatibility.
            input_channels (int): Number of input channels (1 for grayscale).
            dropout_prob (float): Dropout probability.
        """
        super(ConstellationVisionTransformer, self).__init__()
        
        # Load pre-trained ViT model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Modify the first layer for grayscale images
        self.vit.conv_proj = nn.Conv2d(
            1,  # Force single channel input
            768,  # ViT-B/16 hidden size
            kernel_size=16, 
            stride=16
        )
        
        # Freeze the ViT backbone for initial training
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get the hidden size from the ViT model
        hidden_size = self.vit.hidden_dim
        
        # Shared layer after the transformer
        self.shared_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Modulation classification head
        self.modulation_head = nn.Linear(256, num_classes)
        
        # SNR regression head with bounded output
        self.snr_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Bounds output between 0 and 1
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            tuple: (modulation_output, snr_output)
            - modulation_output: class logits [batch_size, num_classes]
            - snr_output: regression values [batch_size, 1] scaled between 0 and 1
        """
        # Get features from the ViT model
        # First, get the patch embedding
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add the cls token
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embeddings
        x = x + self.vit.encoder.pos_embedding
        
        # Apply transformer encoder
        x = self.vit.encoder.layers(x)
        
        # Get the cls token output (first token)
        features = x[:, 0]  # Shape: [batch_size, hidden_size]
        
        # Apply shared layer
        shared_features = self.shared_layer(features)  # Shape: [batch_size, 256]
        
        # Classification heads
        modulation_output = self.modulation_head(shared_features)  # Shape: [batch_size, num_classes]
        
        # SNR regression (bounded between 0 and 1)
        snr_output = self.snr_head(shared_features)  # Shape: [batch_size, 1]
        
        return modulation_output, snr_output
    
    def unfreeze_backbone(self, num_layers=4):
        """
        Unfreeze the last few layers of the ViT backbone for fine-tuning.
        
        Args:
            num_layers (int): Number of transformer layers to unfreeze.
        """
        # Unfreeze the last num_layers transformer blocks
        for i in range(len(self.vit.encoder.layers) - num_layers, len(self.vit.encoder.layers)):
            for param in self.vit.encoder.layers[i].parameters():
                param.requires_grad = True
