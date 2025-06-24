# src/models/swin_transformer_model.py
"""
Swin Transformer implementation for constellation diagram classification.

Based on "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
by Liu et al. (2021) - https://arxiv.org/abs/2103.14030

The Swin Transformer uses hierarchical feature maps and shifted window attention,
making it particularly well-suited for constellation diagrams which exhibit:
1. Sparse spatial structure (most pixels are background)
2. Local patterns that benefit from hierarchical processing
3. Multi-scale features that emerge at different SNR levels

Key advantages for constellation classification:
- Window-based self-attention reduces computational complexity from O(n²) to O(n)
- Hierarchical feature extraction captures both fine-grained and coarse patterns
- Shifted windows enable cross-window connections for global modeling
- Better inductive bias for sparse image data compared to standard ViT

Citation:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
try:
    from torchvision.models import swin_t, swin_s, swin_b
except ImportError:
    # Fallback for older torchvision versions
    print("Warning: Swin models not available in this torchvision version. Install torchvision>=0.13.0")
    swin_t = swin_s = swin_b = None


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


class ConstellationSwinTransformer(nn.Module):
    """
    A Swin Transformer model customized for constellation classification.
    The model outputs two things:
    1) Modulation classification
    2) SNR prediction
    
    Supports Swin-Tiny, Swin-Small, and Swin-Base variants with hierarchical processing
    ideal for constellation diagrams' sparse spatial structure.
    
    The hierarchical attention mechanism is particularly effective for constellation 
    diagrams where signal points cluster at different scales depending on modulation
    type and SNR level.
    """

    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.3, model_variant="swin_tiny"):
        """
        Initialize the ConstellationSwinTransformer model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes.
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.3).
            model_variant (str): Swin variant ('swin_tiny', 'swin_small', 'swin_base'). Defaults to 'swin_tiny'.
        """
        super(ConstellationSwinTransformer, self).__init__()

        # Check if Swin models are available
        if swin_t is None:
            raise ImportError("Swin Transformer models require torchvision>=0.13.0. Please upgrade torchvision.")

        # Select the appropriate Swin model based on variant
        if model_variant == "swin_tiny":
            self.model = swin_t(weights='DEFAULT')
            self.model_name = "swin_tiny"
        elif model_variant == "swin_small":
            self.model = swin_s(weights='DEFAULT')
            self.model_name = "swin_small"
        elif model_variant == "swin_base":
            self.model = swin_b(weights='DEFAULT')
            self.model_name = "swin_base"
        else:
            raise ValueError(f"Unsupported model_variant: {model_variant}. Choose from: swin_tiny, swin_small, swin_base")
        
        self.model_variant = model_variant

        # Modify the input layer to accept the specified number of input channels
        if input_channels != 3:
            # Swin uses features.0.0 as the first conv layer (patch embedding)
            first_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

        # Extract the number of input features to the final classifier
        in_features = self.model.head.in_features

        # Remove the existing classifier head
        self.model.head = nn.Identity()

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
        # Extract hierarchical features using Swin Transformer
        shared_features = self.model(x)

        # Task-specific feature extraction and fusion
        mod_features, snr_features = self.task_specific_extractor(shared_features)

        # Output heads
        modulation_output = self.modulation_head(mod_features)  # Predict modulation class
        snr_output = self.snr_head(snr_features)  # Predict SNR class

        return modulation_output, snr_output