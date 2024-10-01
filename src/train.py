# src/train.py
import torch.nn as nn
import torch.optim as optim
import torch
from models.cnn_model import LightweightCNN
from data_loader import load_data
from visualization import save_all_visualizations
from utils import get_device
from training import train


if __name__ == "__main__":
    # Load data
    print("Loading data...")

    # Load dataset with modulation mapping if needed
    X_train, X_val, X_test, y_train, y_val, y_test, mod2int = load_data()

    # Reshape the input data to include 1 channel for CNN input
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, height, width)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    # Save visualizations for the first 10 samples
    save_all_visualizations(X_train, num_samples=10, output_dir="output")

    # Convert numpy arrays to PyTorch datasets
    train_data = torch.utils.data.TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
    val_data = torch.utils.data.TensorDataset(X_val, torch.tensor(y_val, dtype=torch.long))

    # DataLoader
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = LightweightCNN(num_classes=11)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Determine device (CUDA, MPS, or CPU)
    device = get_device()

    # Train the model
    train(model, device, criterion, optimizer, train_loader, val_loader, epochs=50)
