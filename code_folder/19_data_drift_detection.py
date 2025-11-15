"""
Data Drift Detection
====================
Category 19: MLOps - Detect distribution shifts

Use cases: Monitor production data quality
"""

import numpy as np
from scipy import stats


class DriftDetector:
    """Detect data drift using statistical tests"""
    
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)
    
    def detect_drift(self, new_data, threshold=0.05):
        """Detect drift using Kolmogorov-Smirnov test"""
        drift_detected = []
        p_values = []
        
        for i in range(self.reference_data.shape[1]):
            # KS test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )
            
            p_values.append(p_value)
            drift_detected.append(p_value < threshold)
        
        return {
            'drift_detected': any(drift_detected),
            'features_with_drift': [i for i, d in enumerate(drift_detected) if d],
            'p_values': p_values
        }


def main():
    print("=" * 60)
    print("Data Drift Detection")
    print("=" * 60)
    
    # Reference (training) data
    reference = np.random.randn(1000, 5)
    
    # New data (no drift)
    new_data_no_drift = np.random.randn(100, 5)
    
    # New data (with drift in feature 0)
    new_data_with_drift = np.random.randn(100, 5)
    new_data_with_drift[:, 0] += 2.0  # Shift distribution
    
    detector = DriftDetector(reference)
    
    # Test no drift
    print("\nTesting data without drift:")
    result_no_drift = detector.detect_drift(new_data_no_drift)
    print(f"  Drift detected: {result_no_drift['drift_detected']}")
    
    # Test with drift
    print("\nTesting data with drift:")
    result_with_drift = detector.detect_drift(new_data_with_drift)
    print(f"  Drift detected: {result_with_drift['drift_detected']}")
    print(f"  Features with drift: {result_with_drift['features_with_drift']}")
    
    print("\nKey Takeaways:")
    print("- Data drift degrades model performance")
    print("- Statistical tests (KS, Chi-square) detect shifts")
    print("- Monitor both feature and target drift")
    print("- Trigger model retraining when drift detected")


if __name__ == "__main__":
    main()
