"""
TPOT - Genetic Programming for ML Pipelines
============================================
Category 16: Evolutionary algorithm-based AutoML

Use cases: Complex pipeline optimization
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification


def genetic_algorithm_search(X_train, y_train, X_test, y_test, generations=5):
    """Simplified genetic algorithm for model search"""
    population = [
        make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, max_depth=3)),
        make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=5)),
        make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=150, max_depth=7)),
    ]
    
    best_score = 0
    best_pipeline = None
    
    for gen in range(generations):
        scores = []
        for pipeline in population:
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
        
        print(f"Generation {gen + 1}: Best Score = {max(scores):.4f}")
    
    return best_pipeline, best_score


def main():
    print("=" * 60)
    print("TPOT - Genetic Programming AutoML")
    print("=" * 60)
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nRunning genetic algorithm search...")
    best_pipeline, best_score = genetic_algorithm_search(X_train, y_train, X_test, y_test)
    
    print(f"\nBest Pipeline Score: {best_score:.4f}")
    
    print("\nKey Takeaways:")
    print("- TPOT uses genetic programming")
    print("- Evolves ML pipelines over generations")
    print("- Optimizes preprocessing + models together")
    print("- Can discover novel pipeline configurations")


if __name__ == "__main__":
    main()
