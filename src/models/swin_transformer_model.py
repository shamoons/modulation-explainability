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

    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.3, model_variant="swin_tiny", use_task_specific=False, use_dilated_preprocessing=False):
        """
        Initialize the ConstellationSwinTransformer model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes.
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.3).
            model_variant (str): Swin variant ('swin_tiny', 'swin_small', 'swin_base'). Defaults to 'swin_tiny'.
            use_task_specific (bool): Whether to use task-specific feature extraction (defaults to False).
            use_dilated_preprocessing (bool): Whether to use dilated CNN preprocessing for global context (defaults to False).
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
        self.use_task_specific = use_task_specific
        self.use_dilated_preprocessing = use_dilated_preprocessing
        self.dropout_prob = dropout_prob
        
        # Dilated CNN preprocessing for global constellation context
        if self.use_dilated_preprocessing:
            self.dilated_preprocessing = nn.Sequential(
                # Layer 1: Point detection (RF: 3x3)
                nn.Conv2d(input_channels, 32, kernel_size=3, dilation=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_prob),
                
                # Layer 2: Local clusters (RF: 7x7)
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
            # Update Swin input to expect 3 channels from preprocessing
            swin_input_channels = 3
        else:
            swin_input_channels = input_channels

        # Modify the input layer to accept the specified number of input channels
        if swin_input_channels != 3:
            # Swin uses features.0.0 as the first conv layer (patch embedding)
            first_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                swin_input_channels,
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
            # SNR classification head - outputs num_classes for each SNR class
            self.snr_head = nn.Sequential(
                nn.Linear(in_features // 4, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, snr_classes)  # Output logits for each SNR class
            )
        else:
            # Direct heads without task-specific processing
            self.modulation_head = nn.Linear(in_features, num_classes)
            # SNR classification head - outputs num_classes for each SNR class
            self.snr_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, snr_classes)  # Output logits for each SNR class
            )

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (modulation output, snr output)
        """
        # Apply dilated preprocessing if enabled
        if self.use_dilated_preprocessing:
            x = self.dilated_preprocessing(x)
        
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