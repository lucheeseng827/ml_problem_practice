"""
Feature Engineering Pipeline
=============================
Category 19: MLOps - Production feature engineering

Use cases: Consistent feature computation, data pipelines
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, data):
        """Fit transformers on training data"""
        self.scaler.fit(data)
        self.fitted = True
    
    def transform(self, data):
        """Apply transformations"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        # Scaling
        scaled = self.scaler.transform(data)
        
        # Polynomial features
        poly = np.hstack([scaled, scaled ** 2])
        
        # Interaction features
        interactions = scaled[:, 0:1] * scaled[:, 1:2]
        
        # Combine
        features = np.hstack([poly, interactions])
        
        return features


def main():
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    # Training data
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(20, 2)
    
    # Engineer features
    engineer = FeatureEngineer()
    engineer.fit(X_train)
    
    X_train_features = engineer.transform(X_train)
    X_test_features = engineer.transform(X_test)
    
    print(f"\nOriginal features: {X_train.shape}")
    print(f"Engineered features: {X_train_features.shape}")
    
    print("\nKey Takeaways:")
    print("- Consistent feature engineering train/test")
    print("- Polynomial features capture non-linearity")
    print("- Interaction features model relationships")
    print("- Serialize pipeline for production")


if __name__ == "__main__":
    main()
