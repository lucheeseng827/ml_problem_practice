"""
Autoencoder-based Anomaly Detection
====================================
Category 14: Deep learning for anomaly detection using reconstruction error

Use cases: Manufacturing defects, cybersecurity, medical imaging
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, input_dim=20):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def generate_data():
    np.random.seed(42)
    # Normal samples
    X_normal = np.random.randn(1000, 20)
    # Anomalies (different distribution)
    X_anomaly = np.random.randn(100, 20) * 3 + 5
    X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
    y = np.array([0] * 1000 + [1] * 100)
    return X, y


def main():
    print("=" * 60)
    print("Autoencoder Anomaly Detection")
    print("=" * 60)

    X, y = generate_data()
    X_train = X[y == 0]  # Train only on normal data

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Autoencoder(input_dim=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\nTraining autoencoder on normal data...")
    for epoch in range(50):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            reconstructed = model(x)
            loss = criterion(reconstructed, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/50], Loss: {total_loss / len(loader):.4f}')

    # Detect anomalies using reconstruction error
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        reconstructed = model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

    # Set threshold (95th percentile of normal data errors)
    threshold = np.percentile(reconstruction_errors[:1000], 95)
    predictions = (reconstruction_errors > threshold).astype(int)

    accuracy = np.mean(predictions == y)
    print(f"\nDetection Accuracy: {accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")

    # Visualize reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors[:1000], bins=50, alpha=0.7, label='Normal')
    plt.hist(reconstruction_errors[1000:], bins=50, alpha=0.7, label='Anomaly')
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Autoencoder Reconstruction Errors')
    plt.legend()
    plt.savefig('/tmp/autoencoder_anomaly.png')
    print("Visualization saved to /tmp/autoencoder_anomaly.png")

    print("\nKey Takeaways:")
    print("- Autoencoders learn to reconstruct normal patterns")
    print("- Anomalies have high reconstruction error")
    print("- Threshold determines sensitivity")
    print("- Effective for high-dimensional data")


if __name__ == "__main__":
    main()
