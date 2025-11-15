"""
Voting Ensembles
================
Category 20: Ensemble Methods - Majority Voting & Weighted Voting

This example demonstrates:
- Hard voting (majority vote)
- Soft voting (probability averaging)
- Weighted voting
- Voting classifier for diverse models
- Comparison with individual models

Use cases:
- Combining complementary models
- Reducing variance
- Simple yet effective ensembling
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


def hard_voting_example():
    """Hard voting (majority vote)"""
    print("=" * 60)
    print("Hard Voting Ensemble")
    print("=" * 60)

    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, 3 classes")

    # Define classifiers
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf3 = GaussianNB()

    # Hard voting
    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)],
        voting='hard'  # Majority vote
    )

    print("\nClassifiers:")
    print("  - Logistic Regression")
    print("  - Random Forest")
    print("  - Naive Bayes")

    # Train individual classifiers
    print("\nIndividual classifier performance:")
    for name, clf in [('Logistic Regression', clf1), ('Random Forest', clf2), ('Naive Bayes', clf3)]:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"  {name}: {score:.4f}")

    # Train voting classifier
    print("\nTraining hard voting ensemble...")
    voting_clf.fit(X_train, y_train)
    voting_score = voting_clf.score(X_test, y_test)

    print(f"Hard Voting Ensemble: {voting_score:.4f}")

    return voting_clf


def soft_voting_example():
    """Soft voting (probability averaging)"""
    print("\n" + "=" * 60)
    print("Soft Voting Ensemble")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classifiers with probability estimates
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf3 = SVC(kernel='rbf', probability=True, random_state=42)

    # Hard voting
    hard_voting = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
        voting='hard'
    )

    # Soft voting
    soft_voting = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
        voting='soft'  # Average probabilities
    )

    print("\nTraining both hard and soft voting...")
    hard_voting.fit(X_train, y_train)
    soft_voting.fit(X_train, y_train)

    hard_score = hard_voting.score(X_test, y_test)
    soft_score = soft_voting.score(X_test, y_test)

    print(f"\nHard Voting: {hard_score:.4f}")
    print(f"Soft Voting: {soft_score:.4f}")
    print(f"Difference: {soft_score - hard_score:.4f}")

    # Soft voting usually performs better
    return soft_voting


def weighted_voting_example():
    """Weighted voting (different weights for models)"""
    print("\n" + "=" * 60)
    print("Weighted Voting Ensemble")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = GaussianNB()

    # Equal weights (default)
    equal_voting = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)],
        voting='soft'
    )

    # Weighted voting (higher weight for RF)
    weighted_voting = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)],
        voting='soft',
        weights=[1, 3, 1]  # Give RF 3x weight
    )

    print("\nTraining with different weights...")
    equal_voting.fit(X_train, y_train)
    weighted_voting.fit(X_train, y_train)

    equal_score = equal_voting.score(X_test, y_test)
    weighted_score = weighted_voting.score(X_test, y_test)

    print(f"\nEqual Weights [1, 1, 1]: {equal_score:.4f}")
    print(f"Weighted [1, 3, 1]: {weighted_score:.4f}")

    return weighted_voting


def voting_visualization():
    """Visualize voting behavior on 2D data"""
    print("\n" + "=" * 60)
    print("Voting Decision Boundary Visualization")
    print("=" * 60)

    # Generate 2D data for visualization
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Classifiers
    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf3 = DecisionTreeClassifier(max_depth=3, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)],
        voting='soft'
    )

    # Train all
    models = [
        ('Logistic Regression', clf1),
        ('Random Forest', clf2),
        ('Decision Tree', clf3),
        ('Voting Ensemble', voting_clf)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, clf) in enumerate(models):
        clf.fit(X_train, y_train)

        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
        axes[idx].set_title(f'{name}\nAcc: {clf.score(X_test, y_test):.3f}')

    plt.tight_layout()
    plt.savefig('/tmp/voting_decision_boundaries.png')
    print("\nDecision boundaries saved to /tmp/voting_decision_boundaries.png")


def cross_validation_comparison():
    """Compare voting ensemble with base models using CV"""
    print("\n" + "=" * 60)
    print("Cross-Validation Comparison")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf3 = GradientBoostingClassifier(n_estimators=50, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
        voting='soft'
    )

    print("\nPerforming 5-fold cross-validation...")
    models = [
        ('Logistic Regression', clf1),
        ('Random Forest', clf2),
        ('Gradient Boosting', clf3),
        ('Voting Ensemble', voting_clf)
    ]

    results = []
    for name, clf in models:
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        results.append((name, scores.mean(), scores.std()))
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Visualize
    names, means, stds = zip(*results)

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(names))
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['steelblue', 'steelblue', 'steelblue', 'green'])

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('5-Fold Cross-Validation Comparison')
    plt.xticks(x_pos, names, rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig('/tmp/voting_cv_comparison.png')
    print("\nCV comparison saved to /tmp/voting_cv_comparison.png")


def main():
    """Main execution function"""
    print("Voting Ensembles\n")

    # Example 1: Hard Voting
    hard_voter = hard_voting_example()

    # Example 2: Soft Voting
    soft_voter = soft_voting_example()

    # Example 3: Weighted Voting
    weighted_voter = weighted_voting_example()

    # Example 4: Visualization
    voting_visualization()

    # Example 5: Cross-Validation
    cross_validation_comparison()

    print("\n" + "=" * 60)
    print("Voting Ensembles Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Hard voting: majority vote (classification only)")
    print("- Soft voting: average probabilities (usually better)")
    print("- Weighted voting: give more weight to better models")
    print("- Works best with diverse, uncorrelated models")
    print("- Simple and effective ensembling method")
    print("- Reduces variance through averaging")
    print("- Less prone to overfitting than individual models")
    print("- Computationally efficient (parallel training)")


if __name__ == "__main__":
    main()
