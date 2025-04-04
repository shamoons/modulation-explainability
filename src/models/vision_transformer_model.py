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
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=1536,
            dropout=dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Task-specific transformers
        self.modulation_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.snr_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Cross-task attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.modulation_norm = nn.LayerNorm(self.embed_dim)
        self.snr_norm = nn.LayerNorm(self.embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Classification heads with smaller hidden dimension
        self.modulation_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_classes)
        )
        
        self.snr_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
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
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add task tokens
        modulation_tokens = self.modulation_token.expand(batch_size, -1, -1)
        snr_tokens = self.snr_token.expand(batch_size, -1, -1)
        x = torch.cat([modulation_tokens, snr_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Shared feature extraction
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.shared_transformer(x)
        
        # Split into task-specific tokens and features
        modulation_token = x[:, 0:1]
        snr_token = x[:, 1:2]
        features = x[:, 2:]
        
        # Task-specific processing
        modulation_features = torch.cat([modulation_token, features], dim=1)
        snr_features = torch.cat([snr_token, features], dim=1)
        
        modulation_features = self.modulation_norm(modulation_features)
        snr_features = self.snr_norm(snr_features)
        
        modulation_features = self.modulation_transformer(modulation_features)
        snr_features = self.snr_transformer(snr_features)
        
        # Cross-task attention
        modulation_attended, _ = self.cross_attention(
            modulation_features[:, 0:1],
            snr_features,
            snr_features
        )
        snr_attended, _ = self.cross_attention(
            snr_features[:, 0:1],
            modulation_features,
            modulation_features
        )
        
        # Classification heads
        modulation_output = self.modulation_head(modulation_attended.squeeze(1))
        snr_output = self.snr_head(snr_attended.squeeze(1))
        
        return modulation_output, snr_output
