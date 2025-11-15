"""
Stacking and Blending Ensembles
================================
Category 20: Ensemble Methods - Meta-Learning

This example demonstrates:
- Stacking classifier with multiple base models
- Blending with hold-out set
- Multi-level stacking
- Meta-learner selection
- Cross-validation in stacking

Use cases:
- Kaggle competitions (stacking is very popular)
- Combining diverse models
- Leveraging different algorithm strengths
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


def stacking_classifier_example():
    """Stacking ensemble with multiple base models"""
    print("=" * 60)
    print("Stacking Classifier")
    print("=" * 60)

    # Generate dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Define base models (diverse algorithms)
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]

    # Meta-learner
    meta_learner = LogisticRegression(random_state=42)

    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,  # Cross-validation for generating meta-features
        stack_method='predict_proba',  # Use probabilities as meta-features
        n_jobs=-1
    )

    print("\nBase Models:")
    for name, model in base_models:
        print(f"  - {name}: {model.__class__.__name__}")
    print(f"\nMeta-Learner: {meta_learner.__class__.__name__}")

    # Train individual base models
    print("\nTraining individual base models...")
    base_scores = []
    for name, model in base_models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        base_scores.append((name, score))
        print(f"  {name}: {score:.4f}")

    # Train stacking ensemble
    print("\nTraining stacking ensemble...")
    stacking_clf.fit(X_train, y_train)
    stacking_score = stacking_clf.score(X_test, y_test)

    print(f"\nStacking Ensemble: {stacking_score:.4f}")
    print(f"Improvement over best base: {stacking_score - max([s for _, s in base_scores]):.4f}")

    # Predictions
    y_pred = stacking_clf.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize comparison
    model_names = [name for name, _ in base_scores] + ['Stacking']
    scores = [score for _, score in base_scores] + [stacking_score]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=['steelblue'] * len(base_scores) + ['green'])
    plt.ylabel('Test Accuracy')
    plt.title('Stacking Ensemble vs Base Models')
    plt.ylim(min(scores) - 0.02, max(scores) + 0.02)
    plt.grid(True, axis='y', alpha=0.3)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('/tmp/stacking_comparison.png')
    print("\nComparison plot saved to /tmp/stacking_comparison.png")

    return stacking_clf


def blending_example():
    """Blending with hold-out validation set"""
    print("\n" + "=" * 60)
    print("Blending Ensemble")
    print("=" * 60)

    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)

    # Split into train, validation (for blending), and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Base models
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        SVC(kernel='rbf', probability=True, random_state=42)
    ]

    print("\nTraining base models on training set...")
    # Train on training set
    for model in base_models:
        model.fit(X_train, y_train)

    # Get predictions on validation set (for meta-learner training)
    val_predictions = []
    for model in base_models:
        val_pred_proba = model.predict_proba(X_val)
        val_predictions.append(val_pred_proba)

    # Stack validation predictions
    X_val_meta = np.hstack(val_predictions)

    # Train meta-learner on validation predictions
    print("Training meta-learner (blending)...")
    meta_learner = LogisticRegression(random_state=42)
    meta_learner.fit(X_val_meta, y_val)

    # Get predictions on test set
    test_predictions = []
    for model in base_models:
        test_pred_proba = model.predict_proba(X_test)
        test_predictions.append(test_pred_proba)

    X_test_meta = np.hstack(test_predictions)

    # Final predictions
    y_pred = meta_learner.predict(X_test_meta)
    blending_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nBlending Accuracy: {blending_accuracy:.4f}")

    return meta_learner


def multi_level_stacking():
    """Multi-level stacking (stacking of stacked models)"""
    print("\n" + "=" * 60)
    print("Multi-Level Stacking")
    print("=" * 60)

    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Level 1 base models
    level1_models = [
        ('rf1', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ('rf2', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=43)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
    ]

    # Level 2 meta-model (stacking of level 1)
    level2_meta = LogisticRegression(random_state=42)

    level2_stack = StackingClassifier(
        estimators=level1_models,
        final_estimator=level2_meta,
        cv=3
    )

    # Level 3: Stack the stacked model with additional models
    level3_models = [
        ('stack_l2', level2_stack),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]

    level3_meta = LogisticRegression(random_state=42)

    final_stack = StackingClassifier(
        estimators=level3_models,
        final_estimator=level3_meta,
        cv=3
    )

    print("\nTraining 3-level stacking ensemble...")
    final_stack.fit(X_train, y_train)

    accuracy = final_stack.score(X_test, y_test)
    print(f"Multi-Level Stacking Accuracy: {accuracy:.4f}")

    print("\nArchitecture:")
    print("  Level 1: RF (depth=5), RF (depth=10), GB")
    print("  Level 2: LogisticRegression (meta-learner)")
    print("  Level 3: Level2_Stack + SVM → LogisticRegression")

    return final_stack


def meta_learner_comparison():
    """Compare different meta-learners"""
    print("\n" + "=" * 60)
    print("Meta-Learner Comparison")
    print("=" * 60)

    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ]

    # Different meta-learners
    meta_learners = [
        ('LogisticRegression', LogisticRegression(random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=50, random_state=42))
    ]

    print("\nTesting different meta-learners...")
    results = []

    for meta_name, meta_model in meta_learners:
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3
        )

        stacking_clf.fit(X_train, y_train)
        score = stacking_clf.score(X_test, y_test)
        results.append((meta_name, score))

        print(f"  {meta_name}: {score:.4f}")

    # Visualize
    names, scores = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.barh(names, scores, color='skyblue')
    plt.xlabel('Test Accuracy')
    plt.title('Stacking: Meta-Learner Comparison')
    plt.xlim(min(scores) - 0.01, max(scores) + 0.01)

    for i, (name, score) in enumerate(results):
        plt.text(score, i, f' {score:.4f}', va='center')

    plt.tight_layout()
    plt.savefig('/tmp/meta_learner_comparison.png')
    print("\nMeta-learner comparison saved to /tmp/meta_learner_comparison.png")


def main():
    """Main execution function"""
    print("Stacking and Blending Ensembles\n")

    # Example 1: Stacking
    stacking_model = stacking_classifier_example()

    # Example 2: Blending
    blending_model = blending_example()

    # Example 3: Multi-level Stacking
    multilevel_model = multi_level_stacking()

    # Example 4: Meta-learner Comparison
    meta_learner_comparison()

    print("\n" + "=" * 60)
    print("Stacking & Blending Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Stacking uses cross-validation to generate meta-features")
    print("- Blending uses hold-out set (simpler, less data efficient)")
    print("- Meta-learner learns from base model predictions")
    print("- Use diverse base models for best results")
    print("- Stacking often wins Kaggle competitions")
    print("- Can stack stacked models (multi-level)")
    print("- predict_proba often better than hard predictions")
    print("- Trade-off: complexity vs performance gain")


if __name__ == "__main__":
    main()
