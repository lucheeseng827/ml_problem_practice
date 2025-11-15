"""
Feature Importance Analysis
============================
Category 15: Tree-based feature importance

Use cases: Feature selection, model understanding
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    
    # Plot importances
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].barh(range(20), rf.feature_importances_)
    axes[0].set_title('Random Forest Feature Importance')
    axes[0].set_xlabel('Importance')
    axes[0].set_ylabel('Feature Index')
    
    axes[1].barh(range(20), gb.feature_importances_)
    axes[1].set_title('Gradient Boosting Feature Importance')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('/tmp/feature_importance.png')
    print("\nFeature importance saved to /tmp/feature_importance.png")
    
    print("\nKey Takeaways:")
    print("- Tree models provide built-in feature importance")
    print("- Based on reduction in impurity/gain")
    print("- Helps identify most predictive features")


if __name__ == "__main__":
    main()
