"""
Multiple Outlier Detection Methods (PyOD-style)
================================================
Category 14: Comprehensive anomaly detection comparison

Use cases: Comparing multiple anomaly detection algorithms
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt


def generate_data():
    np.random.seed(42)
    X_normal = np.random.randn(900, 2) * 0.5
    X_anomaly = np.random.uniform(-3, 3, (100, 2))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * 900 + [1] * 100)
    return X, y


def main():
    print("=" * 60)
    print("Multiple Outlier Detection Methods")
    print("=" * 60)

    X, y = generate_data()
    print(f"\nData shape: {X.shape}, Anomalies: {y.sum()}")

    # Multiple detectors
    detectors = {
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
        'One-Class SVM': OneClassSVM(nu=0.1),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True),
        'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42)
    }

    results = {}
    print("\nTraining detectors...")

    for name, detector in detectors.items():
        detector.fit(X)
        predictions = detector.predict(X)
        predictions = (predictions == -1).astype(int)

        auc = roc_auc_score(y, predictions)
        f1 = f1_score(y, predictions)

        results[name] = {'auc': auc, 'f1': f1, 'predictions': predictions}
        print(f"{name}: AUC={auc:.3f}, F1={f1:.3f}")

    # Visualize comparisons
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        pred = result['predictions']
        axes[idx].scatter(X[pred == 0, 0], X[pred == 0, 1], c='blue', label='Normal', alpha=0.6, s=20)
        axes[idx].scatter(X[pred == 1, 0], X[pred == 1, 1], c='red', label='Anomaly', alpha=0.6, s=20)
        axes[idx].set_title(f"{name} (AUC={result['auc']:.3f})")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig('/tmp/outlier_detection_comparison.png')
    print("\nComparison saved to /tmp/outlier_detection_comparison.png")

    print("\nKey Takeaways:")
    print("- Different methods suit different data distributions")
    print("- LOF: density-based, finds local outliers")
    print("- One-Class SVM: finds decision boundary around normal data")
    print("- Elliptic Envelope: assumes Gaussian distribution")
    print("- Ensemble methods often work best")


if __name__ == "__main__":
    main()
