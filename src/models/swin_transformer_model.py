# src/models/swin_transformer_model.py
import torch.nn as nn
import torch
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
    2) SNR prediction
    
    Supports Swin-Tiny, Swin-Small, and Swin-Base variants with hierarchical processing
    ideal for constellation diagrams' sparse spatial structure.
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

        # Shared feature transformation layer
        self.shared_transform = nn.Linear(in_features, in_features // 4)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(in_features // 4)
        self.dropout = nn.Dropout(p=dropout_prob)

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
        features = self.model(x)

        # Apply shared transformation, activation, batch normalization, and dropout
        features = self.relu(self.shared_transform(features))
        features = self.batch_norm(features)
        features = self.dropout(features)

        # Output heads
        modulation_output = self.modulation_head(features)  # Predict modulation class
        snr_output = self.snr_head(features)  # Predict SNR class

        return modulation_output, snr_output