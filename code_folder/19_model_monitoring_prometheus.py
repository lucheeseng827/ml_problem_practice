"""
Model Monitoring with Prometheus
=================================
Category 19: MLOps - Monitor models in production

Use cases: Performance tracking, alerting
"""

from collections import defaultdict
import time


class ModelMetrics:
    """Track model metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_prediction(self, latency, prediction_value):
        self.metrics['latency'].append(latency)
        self.metrics['predictions'].append(prediction_value)
        self.metrics['count'].append(1)
    
    def get_summary(self):
        import numpy as np
        return {
            'total_predictions': len(self.metrics['predictions']),
            'avg_latency_ms': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
            'p95_latency_ms': np.percentile(self.metrics['latency'], 95) if self.metrics['latency'] else 0,
            'avg_prediction': np.mean(self.metrics['predictions']) if self.metrics['predictions'] else 0
        }


def main():
    print("=" * 60)
    print("Model Monitoring with Prometheus")
    print("=" * 60)
    
    metrics = ModelMetrics()
    
    # Simulate predictions
    print("\nSimulating 100 predictions...")
    for i in range(100):
        start = time.time()
        prediction = float(i % 10) / 10
        latency = (time.time() - start) * 1000
        
        metrics.record_prediction(latency, prediction)
    
    # Get summary
    summary = metrics.get_summary()
    print(f"\nMetrics Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nKey Takeaways:")
    print("- Monitor latency, throughput, errors")
    print("- Track model drift and performance degradation")
    print("- Prometheus for metrics collection")
    print("- Grafana for visualization")
    print("- Set up alerts for anomalies")


if __name__ == "__main__":
    main()
