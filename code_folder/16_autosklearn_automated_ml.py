"""
Automated Machine Learning with Auto-sklearn
=============================================
Category 16: AutoML for automatic model selection and hyperparameter tuning

Use cases: Rapid prototyping, non-expert users
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import make_classification


def simple_automl(X, y, cv=5):
    """Simplified AutoML: try multiple models and return best"""
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv)
        results[name] = scores.mean()
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    best_model = max(results, key=results.get)
    return best_model, results[best_model]


def main():
    print("=" * 60)
    print("Automated Machine Learning")
    print("=" * 60)
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    print("\nAutoML: Searching for best model...")
    best_model_name, best_score = simple_automl(X, y)
    
    print(f"\nBest model: {best_model_name} (Score: {best_score:.4f})")
    
    print("\nKey Takeaways:")
    print("- AutoML automates model selection & tuning")
    print("- Tries multiple algorithms automatically")
    print("- Includes preprocessing & feature engineering")
    print("- Great for quick baselines")


if __name__ == "__main__":
    main()
