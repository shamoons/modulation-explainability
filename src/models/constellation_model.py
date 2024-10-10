# src/models/constellation_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCBlock(nn.Module):
    """
    A depthwise separable convolution block.
    Depthwise convolution followed by pointwise convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSCBlock, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class ConstellationCNN(nn.Module):
    def __init__(self, num_classes=11, input_size=(64, 64)):
        super(ConstellationCNN, self).__init__()

        # First convolution block (change in_channels to 1 for grayscale images)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Depthwise separable convolution blocks
        self.dsc_block1 = DSCBlock(32, 64)
        self.dsc_block2 = DSCBlock(64, 128)
        self.dsc_block3 = DSCBlock(128, 256)

        # Global Average Pooling (moved this before _get_conv_output)
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
            dummy_input = torch.ones(1, 1, *input_size)  # Batch size of 1, grayscale (1 channel)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.dsc_block1(x)
            x = self.dsc_block2(x)
            x = self.dsc_block3(x)
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

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Example of creating the model and printing the architecture
    input_size = (64, 64)  # The size of the constellation images
    model = ConstellationCNN(num_classes=11, input_size=input_size)
    print(model)

    # Example of input tensor (Batch size, Channels, Height, Width)
    sample_input = torch.randn(8, 1, *input_size)  # 8 samples, 1 channel (grayscale), 64x64 image size
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
