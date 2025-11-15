"""
Time Series Forecasting with LSTM (Long Short-Term Memory)
===========================================================
Category 10: Time Series Forecasting

This example demonstrates:
- LSTM networks for sequence prediction
- Time series data preparation (windowing)
- Multi-step forecasting
- Bidirectional LSTM

Use cases:
- Stock price prediction
- Energy consumption forecasting
- Traffic flow prediction
- Sensor data analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output from the last time step
        out = self.fc(out[:, -1, :])

        return out


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for improved context understanding"""

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # FC layer (hidden_size * 2 due to bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


def generate_time_series(n_points=1000):
    """Generate synthetic time series with trend and seasonality"""
    t = np.arange(n_points)

    # Multiple components
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50)
    noise = np.random.randn(n_points) * 2

    series = trend + seasonal + noise + 50

    return series


def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def train_lstm_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train LSTM model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    print(f"\nTraining on device: {device}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def lstm_forecasting_example():
    """Complete LSTM time series forecasting example"""
    print("=" * 60)
    print("LSTM Time Series Forecasting")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate data
    print("\nGenerating synthetic time series data...")
    data = generate_time_series(n_points=1000)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Create sequences
    seq_length = 50
    sequences, targets = create_sequences(data_normalized, seq_length)

    # Reshape for LSTM [samples, time_steps, features]
    sequences = sequences.reshape(-1, seq_length, 1)
    targets = targets.reshape(-1, 1)

    # Train/validation/test split
    train_size = int(len(sequences) * 0.7)
    val_size = int(len(sequences) * 0.15)

    train_seq = torch.FloatTensor(sequences[:train_size])
    train_targets = torch.FloatTensor(targets[:train_size])

    val_seq = torch.FloatTensor(sequences[train_size:train_size + val_size])
    val_targets = torch.FloatTensor(targets[train_size:train_size + val_size])

    test_seq = torch.FloatTensor(sequences[train_size + val_size:])
    test_targets = torch.FloatTensor(targets[train_size + val_size:])

    print(f"Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

    # Create data loaders
    train_dataset = TimeSeriesDataset(train_seq, train_targets)
    val_dataset = TimeSeriesDataset(val_seq, val_targets)
    test_dataset = TimeSeriesDataset(test_seq, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = LSTMModel(
        input_size=1,
        hidden_size=50,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ).to(device)

    print(f"\nModel Architecture:")
    print(model)

    # Train model
    print("\nTraining LSTM model...")
    train_losses, val_losses = train_lstm_model(
        model, train_loader, val_loader,
        epochs=50, lr=0.001, device=device
    )

    # Evaluate on test set
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Inverse transform
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)

    print(f"\nTest Set Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Training history
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)

    # Predictions vs Actuals
    axes[1].plot(actuals[:200], label='Actual', alpha=0.7)
    axes[1].plot(predictions[:200], label='Predicted', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Value')
    axes[1].set_title('LSTM Predictions vs Actual (First 200 points)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/lstm_time_series.png')
    print("\nPlot saved to /tmp/lstm_time_series.png")

    # Save model
    torch.save(model.state_dict(), '/tmp/lstm_model.pth')
    print("Model saved to /tmp/lstm_model.pth")

    return model, scaler


def bidirectional_lstm_example():
    """Bidirectional LSTM example"""
    print("\n" + "=" * 60)
    print("Bidirectional LSTM for Time Series")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate and prepare data
    data = generate_time_series(n_points=1000)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    seq_length = 50
    sequences, targets = create_sequences(data_normalized, seq_length)
    sequences = sequences.reshape(-1, seq_length, 1)

    # Split data
    train_size = int(len(sequences) * 0.8)
    train_seq = torch.FloatTensor(sequences[:train_size])
    train_targets = torch.FloatTensor(targets[:train_size])
    val_seq = torch.FloatTensor(sequences[train_size:])
    val_targets = torch.FloatTensor(targets[train_size:])

    train_dataset = TimeSeriesDataset(train_seq, train_targets)
    val_dataset = TimeSeriesDataset(val_seq, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Bidirectional LSTM
    model = BidirectionalLSTM(
        input_size=1,
        hidden_size=50,
        num_layers=2,
        output_size=1
    ).to(device)

    print(f"\nBidirectional LSTM Architecture:")
    print(model)

    # Train
    train_losses, val_losses = train_lstm_model(
        model, train_loader, val_loader,
        epochs=30, lr=0.001, device=device
    )

    print("\nBidirectional LSTM training complete!")
    print("Bidirectional LSTMs process sequences in both directions,")
    print("capturing past and future context for better predictions.")

    return model


def main():
    """Main execution function"""
    print("Time Series Forecasting with PyTorch LSTM\n")

    # Example 1: Standard LSTM
    lstm_model, scaler = lstm_forecasting_example()

    # Example 2: Bidirectional LSTM
    bilstm_model = bidirectional_lstm_example()

    print("\n" + "=" * 60)
    print("LSTM Time Series Forecasting Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- LSTMs excel at learning long-term dependencies")
    print("- Sequence length is a crucial hyperparameter")
    print("- Normalization improves training stability")
    print("- Bidirectional LSTMs can improve accuracy")
    print("- Always validate on unseen data")


if __name__ == "__main__":
    main()
