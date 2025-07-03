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
from .task_specific_extractor import TaskSpecificFeatureExtractor
try:
    from torchvision.models import swin_t, swin_s, swin_b
except ImportError:
    # Fallback for older torchvision versions
    print("Warning: Swin models not available in this torchvision version. Install torchvision>=0.13.0")
    swin_t = swin_s = swin_b = None


class ConstellationSwinTransformer(nn.Module):
    """
    A Swin Transformer model customized for constellation classification.
    The model outputs two things:
    1) Modulation classification
    2) SNR classification
    
    Supports Swin-Tiny, Swin-Small, and Swin-Base variants with hierarchical processing
    ideal for constellation diagrams' sparse spatial structure.
    
    The hierarchical attention mechanism is particularly effective for constellation 
    diagrams where signal points cluster at different scales depending on modulation
    type and SNR level.
    """

    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.3, model_variant="swin_tiny", use_task_specific=False, use_pretrained=True, snr_layer_config="standard"):
        """
        Initialize the ConstellationSwinTransformer model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes.
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.3).
            model_variant (str): Swin variant ('swin_tiny', 'swin_small', 'swin_base'). Defaults to 'swin_tiny'.
            use_task_specific (bool): Whether to use task-specific feature extraction (defaults to False).
            use_pretrained (bool): Whether to use ImageNet pretrained weights (defaults to True).
            snr_layer_config (str): SNR layer configuration ('standard', 'bottleneck_64', 'bottleneck_128', 'dual_layer').
        """
        super(ConstellationSwinTransformer, self).__init__()

        # Check if Swin models are available
        if swin_t is None:
            raise ImportError("Swin Transformer models require torchvision>=0.13.0. Please upgrade torchvision.")

        # Select the appropriate Swin model based on variant
        weights = 'DEFAULT' if use_pretrained else None
        if model_variant == "swin_tiny":
            self.model = swin_t(weights=weights)
            self.model_name = "swin_tiny"
        elif model_variant == "swin_small":
            self.model = swin_s(weights=weights)
            self.model_name = "swin_small"
        elif model_variant == "swin_base":
            self.model = swin_b(weights=weights)
            self.model_name = "swin_base"
        else:
            raise ValueError(f"Unsupported model_variant: {model_variant}. Choose from: swin_tiny, swin_small, swin_base")
        
        self.model_variant = model_variant
        self.use_task_specific = use_task_specific
        self.dropout_prob = dropout_prob

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

        if self.use_task_specific:
            # Task-specific feature extractors that create different representations
            self.task_specific_extractor = TaskSpecificFeatureExtractor(
                input_dim=in_features,
                task_dim=in_features // 4,
                dropout_prob=dropout_prob
            )
            # Output heads for modulation and SNR (reduced feature dimension)
            self.modulation_head = nn.Linear(in_features // 4, num_classes)
            # SNR head configuration
            snr_input_dim = in_features // 4
        else:
            # Direct heads without task-specific processing
            self.modulation_head = nn.Linear(in_features, num_classes)
            # SNR head configuration
            snr_input_dim = in_features
        
        # Configure SNR head based on snr_layer_config
        if snr_layer_config == "standard":
            # Direct linear layer (no bottleneck)
            self.snr_head = nn.Linear(snr_input_dim, snr_classes)
        elif snr_layer_config == "bottleneck_64":
            # 64-dimensional bottleneck
            self.snr_head = nn.Sequential(
                nn.Linear(snr_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(64, snr_classes)
            )
        elif snr_layer_config == "bottleneck_128":
            # 128-dimensional bottleneck
            self.snr_head = nn.Sequential(
                nn.Linear(snr_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(128, snr_classes)
            )
        elif snr_layer_config == "dual_layer":
            # Two-layer architecture with 256->64 compression
            self.snr_head = nn.Sequential(
                nn.Linear(snr_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(64, snr_classes)
            )
        else:
            raise ValueError(f"Unknown snr_layer_config: {snr_layer_config}")

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
        
        # Apply dropout to shared features for regularization
        shared_features = F.dropout(shared_features, p=self.dropout_prob, training=self.training)

        if self.use_task_specific:
            # Task-specific feature extraction and fusion
            mod_features, snr_features = self.task_specific_extractor(shared_features)
            
            # Output heads
            modulation_output = self.modulation_head(mod_features)  # Predict modulation class
            snr_output = self.snr_head(snr_features)  # Predict SNR class
        else:
            # Direct prediction from shared features
            modulation_output = self.modulation_head(shared_features)  # Predict modulation class
            snr_output = self.snr_head(shared_features)  # Predict SNR class

        return modulation_output, snr_output