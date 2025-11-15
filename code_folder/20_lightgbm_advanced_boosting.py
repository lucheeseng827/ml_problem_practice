"""
LightGBM - Advanced Gradient Boosting
======================================
Category 20: Ensemble Methods - High-Performance Gradient Boosting

This example demonstrates:
- LightGBM for fast training on large datasets
- Leaf-wise tree growth strategy
- Categorical feature handling
- GPU acceleration concepts
- Comparison with XGBoost

Use cases:
- Large-scale datasets (millions of samples)
- High-dimensional sparse data
- Faster training than XGBoost
- Production systems requiring speed
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import lightgbm as lgb
import time


def lightgbm_classification_example():
    """LightGBM for classification"""
    print("=" * 60)
    print("LightGBM Classification")
    print("=" * 60)

    # Generate dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    print("\nTraining LightGBM model...")
    start_time = time.time()

    evals_result = {}
    lgb_clf = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        evals_result=evals_result,
        verbose_eval=False
    )

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Predictions
    y_pred_proba = lgb_clf.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nLightGBM Performance:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # Feature Importance
    importance = lgb_clf.feature_importance(importance_type='gain')
    indices = np.argsort(importance)[::-1]

    print(f"\nTop 10 Important Features:")
    for i in range(min(10, len(indices))):
        print(f"  Feature {indices[i]}: {importance[indices[i]]:.4f}")

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['train']['binary_logloss'], label='Train Loss')
    plt.plot(evals_result['test']['binary_logloss'], label='Test Loss')
    plt.xlabel('Boosting Round')
    plt.ylabel('Log Loss')
    plt.title('LightGBM Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/lightgbm_learning_curves.png')
    print("\nLearning curves saved to /tmp/lightgbm_learning_curves.png")

    return lgb_clf


def lightgbm_regression_example():
    """LightGBM for regression"""
    print("\n" + "=" * 60)
    print("LightGBM Regression")
    print("=" * 60)

    # Generate regression data
    X, y = make_regression(
        n_samples=10000,
        n_features=30,
        n_informative=20,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Parameters for regression
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    print("\nTraining LightGBM regressor...")
    lgb_reg = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        valid_names=['test'],
        verbose_eval=False
    )

    # Predictions
    y_pred = lgb_reg.predict(X_test)

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
    plt.title('LightGBM Regression: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('/tmp/lightgbm_regression.png')
    print("Regression plot saved to /tmp/lightgbm_regression.png")

    return lgb_reg


def categorical_features_example():
    """Handle categorical features natively"""
    print("\n" + "=" * 60)
    print("LightGBM with Categorical Features")
    print("=" * 60)

    # Generate mixed data
    np.random.seed(42)
    n_samples = 5000

    # Numerical features
    X_numerical = np.random.randn(n_samples, 10)

    # Categorical features
    X_cat1 = np.random.randint(0, 5, size=(n_samples, 1))  # 5 categories
    X_cat2 = np.random.randint(0, 10, size=(n_samples, 1))  # 10 categories

    X = np.hstack([X_numerical, X_cat1, X_cat2])

    # Target
    y = (X_numerical[:, 0] + X_cat1.squeeze() * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Specify categorical features
    categorical_features = [10, 11]  # Indices of categorical columns

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features
    )
    test_data = lgb.Dataset(
        X_test,
        label=y_test,
        reference=train_data,
        categorical_feature=categorical_features
    )

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }

    print("\nTraining with categorical features...")
    lgb_clf = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        verbose_eval=False
    )

    y_pred = (lgb_clf.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nLightGBM handles categorical features natively without one-hot encoding!")

    return lgb_clf


def dart_boosting_example():
    """DART (Dropouts meet Multiple Additive Regression Trees)"""
    print("\n" + "=" * 60)
    print("LightGBM with DART Boosting")
    print("=" * 60)

    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # DART parameters
    params_dart = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'dart',  # DART boosting
        'num_leaves': 31,
        'learning_rate': 0.05,
        'drop_rate': 0.1,  # Dropout rate
        'skip_drop': 0.5,  # Probability of skipping dropout
        'verbose': -1
    }

    print("\nTraining with DART boosting (adds dropout to trees)...")
    lgb_dart = lgb.train(
        params_dart,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        verbose_eval=False
    )

    y_pred = (lgb_dart.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"DART Test Accuracy: {accuracy:.4f}")
    print("DART reduces overfitting through tree dropout")

    return lgb_dart


def speed_comparison():
    """Compare LightGBM speed with different settings"""
    print("\n" + "=" * 60)
    print("LightGBM Speed Comparison")
    print("=" * 60)

    # Large dataset
    X, y = make_classification(
        n_samples=50000,
        n_features=100,
        n_informative=50,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)

    configs = [
        {'name': 'Default', 'params': {'objective': 'binary', 'verbose': -1}},
        {'name': 'More Leaves', 'params': {'objective': 'binary', 'num_leaves': 63, 'verbose': -1}},
        {'name': 'Histogram (bin=255)', 'params': {'objective': 'binary', 'max_bin': 255, 'verbose': -1}},
    ]

    print("\nTraining speed comparison on 50,000 samples...")
    for config in configs:
        start_time = time.time()

        lgb.train(
            config['params'],
            train_data,
            num_boost_round=100,
            verbose_eval=False
        )

        elapsed = time.time() - start_time
        print(f"  {config['name']}: {elapsed:.2f} seconds")


def hyperparameter_importance():
    """Key hyperparameters in LightGBM"""
    print("\n" + "=" * 60)
    print("LightGBM Key Hyperparameters")
    print("=" * 60)

    print("\nMost Important Hyperparameters:")
    print("\n1. num_leaves (default: 31)")
    print("   - Max number of leaves in one tree")
    print("   - Controls model complexity")
    print("   - Larger = more complex, risk of overfitting")

    print("\n2. learning_rate (default: 0.1)")
    print("   - Shrinkage rate")
    print("   - Lower = slower learning, more robust")
    print("   - Typical range: 0.01 to 0.3")

    print("\n3. n_estimators / num_boost_round")
    print("   - Number of boosting rounds")
    print("   - Use with early stopping")

    print("\n4. max_depth (default: -1, unlimited)")
    print("   - Maximum tree depth")
    print("   - Alternative to num_leaves control")

    print("\n5. min_data_in_leaf (default: 20)")
    print("   - Minimum samples in leaf")
    print("   - Prevents overfitting on small datasets")

    print("\n6. feature_fraction (default: 1.0)")
    print("   - Subsample features")
    print("   - Reduces overfitting, speeds training")

    print("\n7. bagging_fraction (default: 1.0)")
    print("   - Subsample training data")
    print("   - Use with bagging_freq > 0")


def main():
    """Main execution function"""
    print("LightGBM - Advanced Gradient Boosting\n")

    # Example 1: Classification
    lgb_classifier = lightgbm_classification_example()

    # Example 2: Regression
    lgb_regressor = lightgbm_regression_example()

    # Example 3: Categorical Features
    lgb_cat = categorical_features_example()

    # Example 4: DART Boosting
    lgb_dart = dart_boosting_example()

    # Example 5: Speed Comparison
    speed_comparison()

    # Example 6: Hyperparameter Guide
    hyperparameter_importance()

    print("\n" + "=" * 60)
    print("LightGBM Advanced Boosting Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- LightGBM uses leaf-wise tree growth (vs level-wise)")
    print("- Faster training than XGBoost on large datasets")
    print("- Native categorical feature support (no encoding needed)")
    print("- Histogram-based algorithm for speed")
    print("- DART boosting adds dropout for regularization")
    print("- GPU acceleration available")
    print("- Lower memory consumption")
    print("- Excellent for large-scale production systems")
    print("- Popular in Kaggle competitions")


if __name__ == "__main__":
    main()
