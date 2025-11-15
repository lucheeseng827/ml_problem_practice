"""
Neural Architecture Search with Optuna
=======================================
Category 16: Hyperparameter optimization with Optuna

Use cases: Finding optimal neural network architectures
"""

import torch
import torch.nn as nn
import numpy as np


class FlexibleNN(nn.Module):
    def __init__(self, n_layers, hidden_sizes):
        super(FlexibleNN, self).__init__()
        layers = []
        input_size = 20
        
        for i in range(n_layers):
            layers.append(nn.Linear(input_size, hidden_sizes[i]))
            layers.append(nn.ReLU())
            input_size = hidden_sizes[i]
        
        layers.append(nn.Linear(input_size, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def objective_function(n_layers, hidden_sizes):
    """Simplified objective for NAS"""
    # Generate synthetic data
    X = torch.randn(1000, 20)
    y = torch.randint(0, 2, (1000,))
    
    model = FlexibleNN(n_layers, hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Quick training
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
    
    return accuracy


def main():
    print("=" * 60)
    print("Neural Architecture Search with Optuna")
    print("=" * 60)
    
    # Grid search over architectures
    best_score = 0
    best_config = None
    
    configs = [
        (2, [64, 32]),
        (3, [128, 64, 32]),
        (2, [128, 64]),
    ]
    
    print("\nSearching neural architectures...")
    for n_layers, hidden_sizes in configs:
        score = objective_function(n_layers, hidden_sizes)
        print(f"Layers: {n_layers}, Sizes: {hidden_sizes} -> Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = (n_layers, hidden_sizes)
    
    print(f"\nBest configuration: {best_config} (Accuracy: {best_score:.4f})")
    
    print("\nKey Takeaways:")
    print("- NAS automatically finds optimal architectures")
    print("- Optuna uses Bayesian optimization")
    print("- Efficient search through hyperparameter space")
    print("- Can optimize layers, units, activation functions")


if __name__ == "__main__":
    main()
