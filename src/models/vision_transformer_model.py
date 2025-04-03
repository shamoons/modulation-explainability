# src/models/vision_transformer_model.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class ConstellationVisionTransformer(nn.Module):
    """
    A Vision Transformer (ViT) model customized for constellation classification.
    Uses smaller patches (8x8) and task-specific tokens for better handling of constellation diagrams.
    """

    def __init__(self, num_classes=11, snr_classes=26, input_channels=1, dropout_prob=0.5):
        """
        Initialize the ConstellationVisionTransformer model.

        Args:
            num_classes (int): Number of modulation classes.
            snr_classes (int): Number of SNR classes.
            input_channels (int): Number of input channels (1 for grayscale).
            dropout_prob (float): Dropout probability.
        """
        super(ConstellationVisionTransformer, self).__init__()
        
        # Image and patch parameters
        self.image_size = 224
        self.patch_size = 8
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = input_channels * self.patch_size * self.patch_size
        self.embed_dim = 384  # Reduced from 768 to save memory
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            input_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Position embeddings (learned)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 2, self.embed_dim)
        )
        
        # Task-specific tokens
        self.modulation_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.snr_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Transformer encoder with fewer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,  # Reduced from 12
            dim_feedforward=1536,  # Reduced from 3072
            dropout=dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)  # Reduced from 12
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Classification heads with smaller hidden dimension
        self.modulation_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_classes)
        )
        
        self.snr_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, snr_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the model."""
        nn.init.normal_(self.modulation_token, std=0.02)
        nn.init.normal_(self.snr_token, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)
        
        # Initialize patch embedding weights
        nn.init.kaiming_normal_(self.patch_embedding.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            tuple: (modulation_output, snr_output)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Add task tokens
        modulation_tokens = self.modulation_token.expand(batch_size, -1, -1)
        snr_tokens = self.snr_token.expand(batch_size, -1, -1)
        x = torch.cat([modulation_tokens, snr_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Apply transformer
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.transformer(x)
        
        # Extract task-specific tokens
        modulation_token = x[:, 0]  # First token for modulation
        snr_token = x[:, 1]  # Second token for SNR
        
        # Classification heads
        modulation_output = self.modulation_head(modulation_token)
        snr_output = self.snr_head(snr_token)
        
        return modulation_output, snr_output
