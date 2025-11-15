"""
Anomaly Detection with Isolation Forest
========================================
Category 14: Unsupervised anomaly detection

Use cases: Fraud detection, network security, quality control
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def generate_anomaly_data(n_samples=1000, contamination=0.1):
    """Generate data with anomalies"""
    np.random.seed(42)
    # Normal data
    X_normal = np.random.randn(int(n_samples * (1 - contamination)), 2) * 0.5
    # Anomalies
    X_anomaly = np.random.uniform(-4, 4, (int(n_samples * contamination), 2))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * len(X_normal) + [1] * len(X_anomaly))
    return X, y


def main():
    print("=" * 60)
    print("Anomaly Detection with Isolation Forest")
    print("=" * 60)

    # Generate data
    X, y_true = generate_anomaly_data(n_samples=1000, contamination=0.1)
    print(f"\nData shape: {X.shape}")
    print(f"Anomalies: {y_true.sum()} ({y_true.mean():.1%})")

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    y_pred = iso_forest.fit_predict(X)

    # Convert predictions (-1 for anomaly, 1 for normal)
    y_pred = (y_pred == -1).astype(int)

    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    # Anomaly scores
    scores = iso_forest.score_samples(X)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(X[y_true == 0, 0], X[y_true == 0, 1], c='blue', label='Normal', alpha=0.6)
    axes[0].scatter(X[y_true == 1, 0], X[y_true == 1, 1], c='red', label='Anomaly', alpha=0.6)
    axes[0].set_title('True Labels')
    axes[0].legend()

    axes[1].scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', label='Predicted Normal', alpha=0.6)
    axes[1].scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', label='Predicted Anomaly', alpha=0.6)
    axes[1].set_title('Isolation Forest Predictions')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('/tmp/isolation_forest.png')
    print("\nVisualization saved to /tmp/isolation_forest.png")

    print("\nKey Takeaways:")
    print("- Isolation Forest isolates anomalies via random splits")
    print("- Anomalies require fewer splits to isolate")
    print("- Fast and effective for high-dimensional data")
    print("- No need for labeled data (unsupervised)")


if __name__ == "__main__":
    main()
