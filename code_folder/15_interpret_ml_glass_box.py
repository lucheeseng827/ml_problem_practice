"""
Interpretable Glass-Box Models
===============================
Category 15: Inherently interpretable models

Use cases: Regulated industries, high-stakes decisions
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_classification


def main():
    print("=" * 60)
    print("Interpretable Glass-Box Models")
    print("=" * 60)
    
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
    
    # Logistic Regression (interpretable coefficients)
    lr = LogisticRegression()
    lr.fit(X, y)
    
    print("\nLogistic Regression Coefficients:")
    for i, coef in enumerate(lr.coef_[0]):
        print(f"  Feature {i}: {coef:+.4f}")
    
    # Decision Tree (interpretable rules)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)
    
    print("\nDecision Tree Rules:")
    tree_rules = export_text(dt, feature_names=[f'Feature_{i}' for i in range(5)])
    print(tree_rules[:500] + "...")
    
    print("\nKey Takeaways:")
    print("- Glass-box models are inherently interpretable")
    print("- Linear models: simple coefficient interpretation")
    print("- Decision trees: IF-THEN rules")
    print("- Trade-off: interpretability vs performance")


if __name__ == "__main__":
    main()
