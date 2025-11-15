"""
Gradient Boosting with XGBoost
===============================
Category 20: Ensemble Methods - Gradient Boosting Machines

This example demonstrates:
- XGBoost for classification and regression
- Learning rate and tree depth tuning
- Early stopping and regularization
- Feature importance from boosting
- Handling imbalanced data

Use cases:
- Kaggle competitions (most winning solutions)
- Structured/tabular data
- Classification and regression tasks
- Feature engineering evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, mean_squared_error
import xgboost as xgb


def xgboost_classification_example():
    """XGBoost for binary classification"""
    print("=" * 60)
    print("XGBoost Classification")
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
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,  # L1 regularization
        reg_lambda=1,  # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("\nTraining XGBoost classifier...")
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nXGBoost Performance:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # Feature Importance
    importance = xgb_clf.feature_importances_
    indices = np.argsort(importance)[::-1]

    print(f"\nTop 10 Important Features:")
    for i in range(min(10, len(indices))):
        print(f"  Feature {indices[i]}: {importance[indices[i]]:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_clf, max_num_features=15, importance_type='gain')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('/tmp/xgboost_feature_importance.png')
    print("\nFeature importance plot saved to /tmp/xgboost_feature_importance.png")

    # Learning curves
    results = xgb_clf.evals_result()
    train_loss = results['validation_0']['logloss']
    test_loss = results['validation_1']['logloss']

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Boosting Round')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/xgboost_learning_curves.png')
    print("Learning curves saved to /tmp/xgboost_learning_curves.png")

    return xgb_clf


def xgboost_regression_example():
    """XGBoost for regression"""
    print("\n" + "=" * 60)
    print("XGBoost Regression")
    print("=" * 60)

    # Generate regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("\nTraining XGBoost regressor...")
    xgb_reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred = xgb_reg.predict(X_test)

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
    plt.title('XGBoost Regression: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('/tmp/xgboost_regression.png')
    print("Regression plot saved to /tmp/xgboost_regression.png")

    return xgb_reg


def early_stopping_example():
    """Demonstrate early stopping"""
    print("\n" + "=" * 60)
    print("XGBoost with Early Stopping")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000,  # Large number
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("\nTraining with early stopping (patience=10)...")
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )

    print(f"Best iteration: {xgb_clf.best_iteration}")
    print(f"Best score: {xgb_clf.best_score:.4f}")

    accuracy = xgb_clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    return xgb_clf


def learning_rate_comparison():
    """Compare different learning rates"""
    print("\n" + "=" * 60)
    print("Learning Rate Comparison")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    results = {}

    print("\nTraining models with different learning rates...")
    for lr in learning_rates:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=lr,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        xgb_clf.fit(X_train, y_train, verbose=False)
        accuracy = xgb_clf.score(X_test, y_test)

        results[lr] = accuracy
        print(f"  Learning Rate {lr:.2f}: Accuracy = {accuracy:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o', markersize=8)
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('XGBoost: Impact of Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/xgboost_learning_rate.png')
    print("\nLearning rate plot saved to /tmp/xgboost_learning_rate.png")


def regularization_example():
    """Demonstrate L1/L2 regularization"""
    print("\n" + "=" * 60)
    print("XGBoost Regularization")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        {'name': 'No Regularization', 'reg_alpha': 0, 'reg_lambda': 0},
        {'name': 'L2 (lambda=1)', 'reg_alpha': 0, 'reg_lambda': 1},
        {'name': 'L1 (alpha=1)', 'reg_alpha': 1, 'reg_lambda': 0},
        {'name': 'L1+L2', 'reg_alpha': 0.5, 'reg_lambda': 0.5},
    ]

    print("\nComparing regularization strategies...")
    for config in configs:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        xgb_clf.fit(X_train, y_train, verbose=False)
        train_acc = xgb_clf.score(X_train, y_train)
        test_acc = xgb_clf.score(X_test, y_test)

        print(f"\n{config['name']}:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Overfitting: {train_acc - test_acc:.4f}")


def imbalanced_data_example():
    """Handle imbalanced datasets"""
    print("\n" + "=" * 60)
    print("XGBoost with Imbalanced Data")
    print("=" * 60)

    # Create imbalanced dataset (95% class 0, 5% class 1)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        weights=[0.95, 0.05],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nClass distribution:")
    print(f"  Class 0: {np.sum(y_train == 0)} samples")
    print(f"  Class 1: {np.sum(y_train == 1)} samples")

    # Calculate scale_pos_weight
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    # Without balancing
    xgb_clf_no_balance = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf_no_balance.fit(X_train, y_train, verbose=False)

    # With balancing
    xgb_clf_balanced = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf_balanced.fit(X_train, y_train, verbose=False)

    print("\nWithout class balancing:")
    y_pred = xgb_clf_no_balance.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    print("\nWith scale_pos_weight balancing:")
    y_pred_balanced = xgb_clf_balanced.predict(X_test)
    print(classification_report(y_test, y_pred_balanced, target_names=['Class 0', 'Class 1']))


def main():
    """Main execution function"""
    print("Gradient Boosting with XGBoost\n")

    # Example 1: Classification
    xgb_classifier = xgboost_classification_example()

    # Example 2: Regression
    xgb_regressor = xgboost_regression_example()

    # Example 3: Early Stopping
    early_stop_model = early_stopping_example()

    # Example 4: Learning Rate
    learning_rate_comparison()

    # Example 5: Regularization
    regularization_example()

    # Example 6: Imbalanced Data
    imbalanced_data_example()

    print("\n" + "=" * 60)
    print("XGBoost Gradient Boosting Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- XGBoost = Extreme Gradient Boosting")
    print("- Sequential tree building, each corrects previous errors")
    print("- Learning rate controls step size (lower = slower, more robust)")
    print("- Early stopping prevents overfitting")
    print("- Regularization (L1/L2) controls model complexity")
    print("- scale_pos_weight handles class imbalance")
    print("- Feature importance via gain/cover/weight")
    print("- Fast parallel tree construction")
    print("- Dominates Kaggle structured data competitions")


if __name__ == "__main__":
    main()
