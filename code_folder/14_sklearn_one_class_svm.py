"""
One-Class SVM for Novelty Detection
====================================
Category 14: Learning a decision boundary around normal data

Use cases: Novelty detection, one-class classification
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("One-Class SVM for Novelty Detection")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_train = np.random.randn(500, 2) * 0.5
    X_test_normal = np.random.randn(100, 2) * 0.5
    X_test_novel = np.random.uniform(-3, 3, (50, 2))

    X_test = np.vstack([X_test_normal, X_test_novel])
    y_test = np.array([0] * 100 + [1] * 50)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]} (50 novelties)")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train One-Class SVM
    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    svm.fit(X_train_scaled)

    # Predict
    y_pred = svm.predict(X_test_scaled)
    y_pred = (y_pred == -1).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Novel']))

    # Visualize
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = svm.decision_function(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues', alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Training', alpha=0.6, s=20)
    plt.scatter(X_test_normal[:, 0], X_test_normal[:, 1], c='green', label='Test Normal', s=40, edgecolors='k')
    plt.scatter(X_test_novel[:, 0], X_test_novel[:, 1], c='red', label='Novel', s=40, marker='x')
    plt.title('One-Class SVM Decision Boundary')
    plt.legend()
    plt.savefig('/tmp/one_class_svm.png')
    print("Visualization saved to /tmp/one_class_svm.png")

    print("\nKey Takeaways:")
    print("- Learns decision boundary around normal data")
    print("- nu parameter controls outlier fraction")
    print("- RBF kernel captures non-linear patterns")
    print("- Effective for novelty detection")


if __name__ == "__main__":
    main()
