# src/models/vision_transformer_model.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ConstellationVisionTransformer(nn.Module):
    """
    A Vision Transformer (ViT) model for constellation classification.
    Uses a pre-trained ViT model as a base and adds task-specific heads.
    """

    def __init__(self, num_classes=11, dropout_prob=0.3):
        """
        Initialize the ConstellationVisionTransformer model.

        Args:
            num_classes (int): Number of modulation classes.
            dropout_prob (float): Dropout probability.
        """
        super(ConstellationVisionTransformer, self).__init__()
        
        # Shape constants
        # Get hidden dimension and patch size from the ViT model
        self.VIT_HIDDEN_DIM = vit_b_16().hidden_dim  # Get actual hidden dimension from model
        # Get patch size from the ViT model
        self.PATCH_SIZE = vit_b_16().patch_size  # Get actual patch size from model
        self.SHARED_FEATURE_DIM = 256  # Dimension of shared feature layer
        self.SNR_HIDDEN_DIM = 64  # Hidden dimension for SNR head
        self.NUM_ATTENTION_HEADS_MOD = 8  # Number of attention heads for modulation
        self.NUM_ATTENTION_HEADS_SNR = 12  # Number of attention heads for SNR
        
        # Load pre-trained ViT model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Modify the first layer for grayscale images
        self.vit.conv_proj = nn.Conv2d(
            1,  # Force single channel input
            self.VIT_HIDDEN_DIM,
            kernel_size=self.PATCH_SIZE, 
            stride=self.PATCH_SIZE
        )
        
        # Freeze the ViT backbone for initial training
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get the hidden size from the ViT model
        hidden_size = self.vit.hidden_dim
        
        # Task-specific attention layers
        self.mod_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.NUM_ATTENTION_HEADS_MOD,
            dropout=dropout_prob,
            batch_first=True
        )
        
        self.snr_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.NUM_ATTENTION_HEADS_SNR,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Shared layer after the transformer
        self.shared_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.SHARED_FEATURE_DIM),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Modulation classification head
        self.modulation_head = nn.Linear(self.SHARED_FEATURE_DIM, num_classes)
        
        # SNR regression head with bounded output
        self.snr_head = nn.Sequential(
            nn.Linear(self.SHARED_FEATURE_DIM, self.SNR_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(self.SNR_HIDDEN_DIM, 1),
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
        x = self.vit.conv_proj(x)  # Shape: [batch_size, VIT_HIDDEN_DIM, h/PATCH_SIZE, w/PATCH_SIZE]
        x = x.flatten(2).transpose(1, 2)  # Shape: [batch_size, num_patches, VIT_HIDDEN_DIM]
        
        # Add the cls token
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)  # Shape: [batch_size, 1, VIT_HIDDEN_DIM]
        x = torch.cat([cls_token, x], dim=1)  # Shape: [batch_size, num_patches + 1, VIT_HIDDEN_DIM]
        
        # Add position embeddings
        x = x + self.vit.encoder.pos_embedding  # Shape: [batch_size, num_patches + 1, VIT_HIDDEN_DIM]
        
        # Apply transformer encoder
        features = self.vit.encoder.layers(x)  # Shape: [batch_size, num_patches + 1, VIT_HIDDEN_DIM]
        
        # Apply task-specific attention
        mod_features, _ = self.mod_attention(
            features, features, features
        )
        
        snr_features, _ = self.snr_attention(
            features, features, features
        )  # Shape: [batch_size, num_patches + 1, VIT_HIDDEN_DIM]
        
        # Get cls token outputs (first token)
        mod_features = mod_features[:, 0]  # Shape: [batch_size, VIT_HIDDEN_DIM]
        snr_features = snr_features[:, 0]  # Shape: [batch_size, VIT_HIDDEN_DIM]
        
        # Apply shared layer to both features
        mod_shared = self.shared_layer(mod_features)  # Shape: [batch_size, SHARED_FEATURE_DIM]
        snr_shared = self.shared_layer(snr_features)  # Shape: [batch_size, SHARED_FEATURE_DIM]
        
        # Classification heads
        modulation_output = self.modulation_head(mod_shared)  # Shape: [batch_size, num_classes]
        snr_output = self.snr_head(snr_shared)  # Shape: [batch_size, 1]
        
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
