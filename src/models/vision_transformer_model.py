# src/models/vision_transformer_model.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import vit_b_16, vit_b_32


class TaskSpecificFeatureExtractor(nn.Module):
    """
    Task-specific feature extraction that creates different representations for each task
    before applying task heads. This prevents task competition at the feature level.
    
    Uses different transformations and attention mechanisms to create task-specific
    feature spaces, then applies residual connections and gating to preserve information.
    """
    
    def __init__(self, input_dim, task_dim, dropout_prob=0.3):
        super().__init__()
        
        # Different activation functions to create different feature distributions
        self.mod_branch = nn.Sequential(
            nn.Linear(input_dim, task_dim),
            nn.GELU(),  # Different activation for modulation task
            nn.BatchNorm1d(task_dim),
            nn.Dropout(dropout_prob)
        )
        
        self.snr_branch = nn.Sequential(
            nn.Linear(input_dim, task_dim),
            nn.ReLU(),  # Different activation for SNR task
            nn.BatchNorm1d(task_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Task-specific attention mechanisms
        self.mod_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.ReLU(),
            nn.Linear(input_dim // 8, input_dim),
            nn.Sigmoid()
        )
        
        self.snr_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.Tanh(),  # Different activation
            nn.Linear(input_dim // 8, input_dim),
            nn.Sigmoid()
        )
        
        # Residual projections to preserve information
        self.mod_residual = nn.Linear(input_dim, task_dim) if input_dim != task_dim else nn.Identity()
        self.snr_residual = nn.Linear(input_dim, task_dim) if input_dim != task_dim else nn.Identity()
        
    def forward(self, shared_features):
        """
        Extract task-specific features using different attention patterns and transformations.
        
        Args:
            shared_features: Features from the backbone model
            
        Returns:
            tuple: (modulation_features, snr_features)
        """
        # Apply task-specific attention to shared features
        mod_attended = shared_features * self.mod_attention(shared_features)
        snr_attended = shared_features * self.snr_attention(shared_features)
        
        # Transform through task-specific branches
        mod_features = self.mod_branch(mod_attended)
        snr_features = self.snr_branch(snr_attended)
        
        # Add residual connections to preserve information
        mod_residual = self.mod_residual(shared_features)
        snr_residual = self.snr_residual(shared_features)
        
        # Weighted combination instead of simple addition
        mod_features = 0.7 * mod_features + 0.3 * mod_residual
        snr_features = 0.7 * snr_features + 0.3 * snr_residual
        
        return mod_features, snr_features


class ConstellationVisionTransformer(nn.Module):
    """
    A simplified wrapper around a Vision Transformer (ViT) model from torchvision, customized for constellation classification.
    The model outputs two things:
    1) Modulation classification
    2) SNR prediction
    
    Supports both ViT-B/16 (patch_size=16) and ViT-B/32 (patch_size=32) variants.
    """

    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.3, patch_size=16):
        """
        Initialize the ConstellationVisionTransformer model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes.
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.3).
            patch_size (int): Patch size for ViT (16 or 32). Defaults to 16.
        """
        super(ConstellationVisionTransformer, self).__init__()

        # Select the appropriate ViT model based on patch size
        if patch_size == 16:
            self.model = vit_b_16(weights='DEFAULT')
            self.model_name = "vit_b_16"
        elif patch_size == 32:
            self.model = vit_b_32(weights='DEFAULT')
            self.model_name = "vit_b_32"
        else:
            raise ValueError(f"Unsupported patch_size: {patch_size}. Choose 16 or 32.")
        
        self.patch_size = patch_size

        # Modify the input embedding layer to accept the specified number of input channels
        if input_channels != 3:
            conv_proj = self.model.conv_proj
            self.model.conv_proj = nn.Conv2d(
                input_channels,
                conv_proj.out_channels,
                kernel_size=conv_proj.kernel_size,
                stride=conv_proj.stride,
                padding=conv_proj.padding,
                bias=conv_proj.bias is not None,
            )

        # Extract the number of input features to the final fully connected (fc) layer
        in_features = self.model.heads.head.in_features

        # Remove the existing fully connected layer
        self.model.heads.head = nn.Identity()

        # Task-specific feature extractors that create different representations
        self.task_specific_extractor = TaskSpecificFeatureExtractor(
            input_dim=in_features,
            task_dim=in_features // 4,
            dropout_prob=dropout_prob
        )

        # Output heads for modulation and SNR
        self.modulation_head = nn.Linear(in_features // 4, num_classes)
        self.snr_head = nn.Linear(in_features // 4, snr_classes)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (modulation output, snr output)
        """
        # Extract features using Vision Transformer
        shared_features = self.model(x)

        # Task-specific feature extraction and fusion
        mod_features, snr_features = self.task_specific_extractor(shared_features)

        # Output heads
        modulation_output = self.modulation_head(mod_features)  # Predict modulation class
        snr_output = self.snr_head(snr_features)  # Predict SNR class

        return modulation_output, snr_output
