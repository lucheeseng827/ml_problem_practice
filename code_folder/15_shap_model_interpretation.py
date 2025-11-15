"""
Model Interpretation with SHAP
===============================
Category 15: Explain model predictions using SHAP values

Use cases: Model debugging, regulatory compliance, trust building
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SimpleSHAP:
    """Simplified SHAP-like feature importance"""
    
    def __init__(self, model, X_background):
        self.model = model
        self.X_background = X_background
        
    def explain(self, X_instance):
        """Calculate feature contributions"""
        base_prediction = self.model.predict_proba(self.X_background).mean(axis=0)
        instance_prediction = self.model.predict_proba([X_instance])[0]
        
        contributions = np.zeros(len(X_instance))
        for i in range(len(X_instance)):
            X_permuted = self.X_background.copy()
            X_permuted[:, i] = X_instance[i]
            perm_prediction = self.model.predict_proba(X_permuted).mean(axis=0)
            contributions[i] = (perm_prediction - base_prediction)[1]
            
        return contributions


def main():
    print("=" * 60)
    print("Model Interpretation with SHAP")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nModel accuracy: {model.score(X_test, y_test):.4f}")
    
    # SHAP explanation
    explainer = SimpleSHAP(model, X_train[:100])
    instance = X_test[0]
    contributions = explainer.explain(instance)
    
    print(f"\nFeature contributions for sample prediction:")
    for i, contrib in enumerate(contributions):
        print(f"  Feature {i}: {contrib:+.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in contributions]
    plt.barh(range(len(contributions)), contributions, color=colors)
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature Index')
    plt.title('Feature Contributions (SHAP-like)')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('/tmp/shap_interpretation.png')
    print("\nVisualization saved to /tmp/shap_interpretation.png")
    
    print("\nKey Takeaways:")
    print("- SHAP values explain individual predictions")
    print("- Based on game theory (Shapley values)")
    print("- Model-agnostic interpretation")
    print("- Positive/negative feature contributions")
    print("- Helps build trust in ML models")


if __name__ == "__main__":
    main()
