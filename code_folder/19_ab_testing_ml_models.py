"""
A/B Testing ML Models
======================
Category 19: MLOps - Compare model variants in production

Use cases: Model selection, gradual rollouts
"""

import numpy as np


class ABTest:
    """A/B test two models"""
    
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results_a = []
        self.results_b = []
    
    def route_request(self, features):
        """Route to model A or B"""
        if np.random.rand() < self.traffic_split:
            result = self.model_a(features)
            self.results_a.append(result)
            return 'A', result
        else:
            result = self.model_b(features)
            self.results_b.append(result)
            return 'B', result
    
    def analyze(self):
        """Analyze A/B test results"""
        return {
            'model_a_requests': len(self.results_a),
            'model_b_requests': len(self.results_b),
            'model_a_avg': np.mean(self.results_a) if self.results_a else 0,
            'model_b_avg': np.mean(self.results_b) if self.results_b else 0
        }


def model_a(features):
    return np.sum(features) * 1.0

def model_b(features):
    return np.sum(features) * 1.1


def main():
    print("=" * 60)
    print("A/B Testing ML Models")
    print("=" * 60)
    
    ab_test = ABTest(model_a, model_b, traffic_split=0.5)
    
    # Simulate requests
    print("\nRunning A/B test with 1000 requests...")
    for _ in range(1000):
        features = np.random.randn(5)
        variant, result = ab_test.route_request(features)
    
    # Analyze
    results = ab_test.analyze()
    print(f"\nA/B Test Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nKey Takeaways:")
    print("- A/B tests compare model variants")
    print("- Split traffic between models")
    print("- Measure business metrics (CTR, revenue, etc.)")
    print("- Statistical significance testing")
    print("- Gradual rollouts reduce risk")


if __name__ == "__main__":
    main()
