"""
AdaBoost - Adaptive Boosting
=============================
Category 20: Ensemble Methods - Sequential Weak Learner Boosting

This example demonstrates:
- AdaBoost for classification and regression
- Weak learner concept (shallow trees)
- Sample weight adaptation
- Learning curves and convergence
- Comparison with strong learners

Use cases:
- Binary and multi-class classification
- When interpretability is important
- Relatively clean datasets
- Face detection (Viola-Jones algorithm)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression, make_moons
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error


def adaboost_classification_example():
    """AdaBoost for binary classification"""
    print("=" * 60)
    print("AdaBoost Classification")
    print("=" * 60)

    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

    # AdaBoost with decision stumps
    ada_clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME.R',
        random_state=42
    )

    print("\nTraining AdaBoost with decision stumps...")
    ada_clf.fit(X_train, y_train)

    # Predictions
    y_pred = ada_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAdaBoost Performance:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Estimator weights
    estimator_weights = ada_clf.estimator_weights_
    print(f"\nEstimator weights (first 10): {estimator_weights[:10]}")

    # Plot estimator weights
    plt.figure(figsize=(10, 6))
    plt.plot(estimator_weights)
    plt.xlabel('Estimator Index')
    plt.ylabel('Weight')
    plt.title('AdaBoost Estimator Weights')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/adaboost_weights.png')
    print("Estimator weights plot saved to /tmp/adaboost_weights.png")

    return ada_clf


def weak_vs_strong_learner():
    """Compare weak learners (stumps) vs strong learners (deep trees)"""
    print("\n" + "=" * 60)
    print("Weak vs Strong Learners in AdaBoost")
    print("=" * 60)

    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Weak learner (stump)
    ada_weak = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )

    # Strong learner (deep tree)
    ada_strong = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        random_state=42
    )

    print("\nTraining AdaBoost with weak learners...")
    ada_weak.fit(X_train, y_train)
    weak_accuracy = ada_weak.score(X_test, y_test)

    print("Training AdaBoost with strong learners...")
    ada_strong.fit(X_train, y_train)
    strong_accuracy = ada_strong.score(X_test, y_test)

    print(f"\nWeak Learners (max_depth=1): {weak_accuracy:.4f}")
    print(f"Strong Learners (max_depth=5): {strong_accuracy:.4f}")

    # Visualize decision boundaries
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, title) in enumerate([(ada_weak, 'Weak Learners'), (ada_strong, 'Strong Learners')]):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
        axes[idx].set_title(f'AdaBoost with {title}')

    plt.tight_layout()
    plt.savefig('/tmp/adaboost_decision_boundaries.png')
    print("\nDecision boundaries saved to /tmp/adaboost_decision_boundaries.png")


def adaboost_regression_example():
    """AdaBoost for regression"""
    print("\n" + "=" * 60)
    print("AdaBoost Regression")
    print("=" * 60)

    # Generate regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # AdaBoost Regressor
    ada_reg = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=50,
        learning_rate=1.0,
        loss='linear',  # 'linear', 'square', or 'exponential'
        random_state=42
    )

    print("\nTraining AdaBoost regressor...")
    ada_reg.fit(X_train, y_train)

    # Predictions
    y_pred = ada_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nRegression Performance:")
    print(f"RMSE: {rmse:.4f}")

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('AdaBoost Regression: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('/tmp/adaboost_regression.png')
    print("Regression plot saved to /tmp/adaboost_regression.png")

    return ada_reg


def learning_rate_analysis():
    """Analyze impact of learning rate"""
    print("\n" + "=" * 60)
    print("Learning Rate Analysis")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    learning_rates = [0.1, 0.5, 1.0, 1.5, 2.0]
    results = []

    print("\nTesting different learning rates...")
    for lr in learning_rates:
        ada_clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=lr,
            random_state=42
        )

        ada_clf.fit(X_train, y_train)
        train_acc = ada_clf.score(X_train, y_train)
        test_acc = ada_clf.score(X_test, y_test)

        results.append((lr, train_acc, test_acc))
        print(f"  LR={lr:.1f}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    # Plot
    results = np.array(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results[:, 0], results[:, 1], marker='o', label='Train Accuracy')
    plt.plot(results[:, 0], results[:, 2], marker='s', label='Test Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('AdaBoost: Impact of Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/adaboost_learning_rate.png')
    print("\nLearning rate analysis saved to /tmp/adaboost_learning_rate.png")


def convergence_analysis():
    """Analyze convergence with number of estimators"""
    print("\n" + "=" * 60)
    print("Convergence Analysis")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators_range = range(1, 101, 5)
    train_scores = []
    test_scores = []

    print("\nTraining AdaBoost with varying n_estimators...")
    for n_est in n_estimators_range:
        ada_clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_est,
            random_state=42
        )

        ada_clf.fit(X_train, y_train)
        train_scores.append(ada_clf.score(X_train, y_train))
        test_scores.append(ada_clf.score(X_test, y_test))

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, marker='o', label='Train Score')
    plt.plot(n_estimators_range, test_scores, marker='s', label='Test Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('AdaBoost Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/adaboost_convergence.png')
    print("Convergence plot saved to /tmp/adaboost_convergence.png")


def main():
    """Main execution function"""
    print("AdaBoost - Adaptive Boosting\n")

    # Example 1: Classification
    ada_classifier = adaboost_classification_example()

    # Example 2: Weak vs Strong Learners
    weak_vs_strong_learner()

    # Example 3: Regression
    ada_regressor = adaboost_regression_example()

    # Example 4: Learning Rate
    learning_rate_analysis()

    # Example 5: Convergence
    convergence_analysis()

    print("\n" + "=" * 60)
    print("AdaBoost Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- AdaBoost focuses on misclassified samples")
    print("- Increases weights of incorrectly predicted samples")
    print("- Combines weak learners into strong ensemble")
    print("- Decision stumps (depth=1) are common weak learners")
    print("- SAMME.R uses probability estimates (better than SAMME)")
    print("- Learning rate controls contribution of each estimator")
    print("- Sensitive to noisy data and outliers")
    print("- Historically important (Viola-Jones face detection)")


if __name__ == "__main__":
    main()
