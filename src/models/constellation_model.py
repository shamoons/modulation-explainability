# src/models/constellation_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph


class ResidualDSCBlock(nn.Module):
    """
    A depthwise separable convolution block with residual connection.
    Depthwise convolution followed by pointwise convolution, with a residual (skip) connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualDSCBlock, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Shortcut connection for residual (in case in_channels != out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # Create shortcut connection
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return F.relu(out + residual)  # Add residual connection and apply ReLU


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block to apply channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze step
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = self.global_avg_pool(x).view(batch_size, channels)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(batch_size, channels, 1, 1)
        return x * se.expand_as(x)  # Apply attention (excitation)


class ConstellationCNN(nn.Module):
    def __init__(self, num_classes=11, input_size=(224, 224)):
        super(ConstellationCNN, self).__init__()

        # Modify first convolution block to accept 3 channels for the input image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Changed in_channels to 3
        self.bn1 = nn.BatchNorm2d(32)

        # Residual Depthwise Separable Convolution blocks
        self.dsc_block1 = ResidualDSCBlock(32, 64)
        self.dsc_block2 = ResidualDSCBlock(64, 128)
        self.dsc_block3 = ResidualDSCBlock(128, 256)

        # Add Squeeze-and-Excitation (SE) Block after convolution blocks
        self.se_block = SEBlock(256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Compute the size after convolution and pooling
        self.feature_size = self._get_conv_output(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, input_size):
        """
        Helper function to compute the size of the features after the convolution layers.
        This helps to dynamically calculate the input size for the fully connected layer.
        """
        with torch.no_grad():
            dummy_input = torch.ones(1, 3, *input_size)  # Batch size of 1, 3 channels for the input image
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.dsc_block1(x)
            x = self.dsc_block2(x)
            x = self.dsc_block3(x)
            x = self.se_block(x)  # Apply SE block here
            x = self.global_avg_pool(x)
            feature_size = x.view(1, -1).size(1)
        return feature_size

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Pass through depthwise separable convolution blocks
        x = self.dsc_block1(x)
        x = self.dsc_block2(x)
        x = self.dsc_block3(x)

        # Apply Squeeze-and-Excitation block
        x = self.se_block(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Example of creating the model and printing the architecture
    input_size = (224, 224)  # The size of the constellation images
    model = ConstellationCNN(num_classes=24, input_size=input_size)
    print(model)

    # Example of input tensor (Batch size, Channels, Height, Width)
    sample_input = torch.randn(1, 3, *input_size)  # 1 sample, 3 channels, 64x64 image size

    # Create a visualization of the model's computation graph using torchview
    model_graph = draw_graph(model, input_size=(1, 3, *input_size))

    # Save the graph as a PNG file
    model_graph.visual_graph.render("constellation_model_torchview_graph", format="png")
