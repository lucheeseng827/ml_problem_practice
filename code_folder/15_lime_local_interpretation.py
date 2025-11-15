"""
LIME - Local Interpretable Model-Agnostic Explanations
=======================================================
Category 15: Explain individual predictions locally

Use cases: Understanding specific predictions, debugging models
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression


class SimpleLIME:
    """Simplified LIME implementation"""
    
    def __init__(self, model, n_samples=1000):
        self.model = model
        self.n_samples = n_samples
        
    def explain(self, instance, X_train):
        # Generate perturbed samples around instance
        perturbations = instance + np.random.randn(self.n_samples, len(instance)) * 0.5
        
        # Get model predictions for perturbations
        predictions = self.model.predict_proba(perturbations)[:, 1]
        
        # Fit linear model locally
        local_model = LinearRegression()
        local_model.fit(perturbations, predictions)
        
        return local_model.coef_


def main():
    print("=" * 60)
    print("LIME - Local Model Interpretation")
    print("=" * 60)
    
    # Data and model
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Explain instance
    instance = X[0]
    lime = SimpleLIME(model)
    importance = lime.explain(instance, X)
    
    print("\nLocal feature importance:")
    for i, imp in enumerate(importance):
        print(f"  Feature {i}: {imp:+.4f}")
    
    print("\nKey Takeaways:")
    print("- LIME explains individual predictions locally")
    print("- Fits interpretable model around instance")
    print("- Model-agnostic approach")
    print("- Works for any black-box model")


if __name__ == "__main__":
    main()
