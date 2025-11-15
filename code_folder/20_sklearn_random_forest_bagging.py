"""
Random Forest and Bagging Ensembles
====================================
Category 20: Ensemble Methods - Bootstrap Aggregating (Bagging)

This example demonstrates:
- Random Forest classifier and regressor
- Bagging with decision trees
- Feature importance from ensembles
- Out-of-bag (OOB) error estimation
- Hyperparameter tuning for RF

Use cases:
- General-purpose classification/regression
- Feature selection
- Handling high-dimensional data
- Robust predictions with variance reduction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import pandas as pd


def random_forest_classification_example():
    """Comprehensive Random Forest classification"""
    print("=" * 60)
    print("Random Forest Classification")
    print("=" * 60)

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRandom Forest Performance:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = rf.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    print(f"\nTop 10 Most Important Features:")
    for i in range(min(10, len(indices))):
        print(f"  Feature {indices[i]}: {feature_importance[indices[i]]:.4f}")

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), feature_importance[indices])
    plt.xlabel('Feature Index (sorted by importance)')
    plt.ylabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('/tmp/rf_feature_importance.png')
    print("\nFeature importance plot saved to /tmp/rf_feature_importance.png")

    return rf


def bagging_classifier_example():
    """Bagging with decision trees"""
    print("\n" + "=" * 60)
    print("Bagging Classifier with Decision Trees")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Single decision tree
    single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    single_tree.fit(X_train, y_train)
    single_accuracy = single_tree.score(X_test, y_test)

    # Bagging ensemble
    bagging = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    bagging.fit(X_train, y_train)
    bagging_accuracy = bagging.score(X_test, y_test)

    print(f"\nSingle Decision Tree Accuracy: {single_accuracy:.4f}")
    print(f"Bagging Ensemble Accuracy: {bagging_accuracy:.4f}")
    print(f"OOB Score: {bagging.oob_score_:.4f}")
    print(f"Improvement: {(bagging_accuracy - single_accuracy):.4f}")

    return bagging


def random_forest_regression_example():
    """Random Forest for regression tasks"""
    print("\n" + "=" * 60)
    print("Random Forest Regression")
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

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRegression Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"OOB Score: {rf_reg.oob_score_:.4f}")

    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('/tmp/rf_regression_predictions.png')
    print("Prediction plot saved to /tmp/rf_regression_predictions.png")

    return rf_reg


def hyperparameter_tuning_example():
    """Grid search for Random Forest hyperparameters"""
    print("\n" + "=" * 60)
    print("Random Forest Hyperparameter Tuning")
    print("=" * 60)

    X, y = make_classification(n_samples=500, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42)

    print("\nPerforming grid search...")
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    # Test set performance
    best_rf = grid_search.best_estimator_
    test_score = best_rf.score(X_test, y_test)
    print(f"Test Set Accuracy: {test_score:.4f}")

    # Results DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]

    print("\nTop 5 Configurations:")
    print(top_5.to_string(index=False))

    return best_rf


def estimator_analysis():
    """Analyze impact of number of estimators"""
    print("\n" + "=" * 60)
    print("Number of Estimators Analysis")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators_range = [10, 25, 50, 100, 200, 300]
    train_scores = []
    test_scores = []
    oob_scores = []

    print("\nTraining Random Forests with different n_estimators...")
    for n_est in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=10,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
        oob_scores.append(rf.oob_score_)

        print(f"  n_estimators={n_est:3d}: Train={train_scores[-1]:.4f}, "
              f"Test={test_scores[-1]:.4f}, OOB={oob_scores[-1]:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, marker='o', label='Train Score')
    plt.plot(n_estimators_range, test_scores, marker='s', label='Test Score')
    plt.plot(n_estimators_range, oob_scores, marker='^', label='OOB Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest: Impact of Number of Estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/rf_n_estimators_analysis.png')
    print("\nEstimators analysis plot saved to /tmp/rf_n_estimators_analysis.png")


def main():
    """Main execution function"""
    print("Random Forest and Bagging Ensembles\n")

    # Example 1: Random Forest Classification
    rf_classifier = random_forest_classification_example()

    # Example 2: Bagging Classifier
    bagging_clf = bagging_classifier_example()

    # Example 3: Random Forest Regression
    rf_regressor = random_forest_regression_example()

    # Example 4: Hyperparameter Tuning
    tuned_rf = hyperparameter_tuning_example()

    # Example 5: Estimator Analysis
    estimator_analysis()

    print("\n" + "=" * 60)
    print("Random Forest & Bagging Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Random Forest = Bagging + Random Feature Selection")
    print("- Bootstrap sampling reduces variance")
    print("- Feature randomness decorrelates trees")
    print("- OOB score provides unbiased error estimate")
    print("- More trees → more stable, diminishing returns after ~100-200")
    print("- Feature importance identifies predictive features")
    print("- Parallel training makes RF scalable")
    print("- Works well with high-dimensional data")
    print("- Resistant to overfitting with proper tuning")


if __name__ == "__main__":
    main()
