"""
Model Production Validation & Battle Testing
=============================================
Category 19: MLOps - Pre-deployment validation for production readiness

Use cases: Model validation gates, TPS/throughput assurance on EKS,
           latency SLA checks, accuracy regression detection, load testing
Demonstrates: Production readiness checks before Argo Rollouts promotion
"""

import json
import os
import tempfile
import time
import threading
import statistics
import concurrent.futures
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# vCPU and memory for common EKS instance types used in ML serving
INSTANCE_SPECS: dict = {
    "m5.large":    {"vcpus": 2,  "memory_gib": 8},
    "m5.xlarge":   {"vcpus": 4,  "memory_gib": 16},
    "m5.2xlarge":  {"vcpus": 8,  "memory_gib": 32},
    "m5.4xlarge":  {"vcpus": 16, "memory_gib": 64},
    "m5.8xlarge":  {"vcpus": 32, "memory_gib": 128},
    "c5.xlarge":   {"vcpus": 4,  "memory_gib": 8},
    "c5.2xlarge":  {"vcpus": 8,  "memory_gib": 16},
    "c5.4xlarge":  {"vcpus": 16, "memory_gib": 32},
    "r5.xlarge":   {"vcpus": 4,  "memory_gib": 32},
    "r5.2xlarge":  {"vcpus": 8,  "memory_gib": 64},
    "g4dn.xlarge": {"vcpus": 4,  "memory_gib": 16},
    "g4dn.2xlarge": {"vcpus": 8, "memory_gib": 32},
    "g5.xlarge":   {"vcpus": 4,  "memory_gib": 16},
    "g5.2xlarge":  {"vcpus": 8,  "memory_gib": 32},
    "p3.2xlarge":  {"vcpus": 8,  "memory_gib": 61},
}


@dataclass
class EKSClusterSpec:
    """Known EKS cluster specification used for capacity planning."""
    cluster_name: str = "ml-serving-prod"
    node_instance_type: str = "m5.2xlarge"
    node_count: int = 3
    gpu_instance_type: Optional[str] = "g4dn.xlarge"
    gpu_node_count: int = 0
    max_pods_per_node: int = 58  # ENI-based limit for m5.2xlarge
    target_cpu_utilization: float = 0.70
    target_memory_utilization: float = 0.75

    def _get_instance_spec(self) -> dict:
        if self.node_instance_type not in INSTANCE_SPECS:
            raise ValueError(
                f"Unknown instance type '{self.node_instance_type}'. "
                f"Add it to INSTANCE_SPECS or use one of: "
                f"{', '.join(sorted(INSTANCE_SPECS))}"
            )
        return INSTANCE_SPECS[self.node_instance_type]

    @property
    def total_vcpus(self) -> int:
        return self.node_count * self._get_instance_spec()["vcpus"]

    @property
    def total_memory_gib(self) -> int:
        return self.node_count * self._get_instance_spec()["memory_gib"]


@dataclass
class ValidationThresholds:
    """Thresholds a model must meet to be promoted to production."""
    min_accuracy: float = 0.85
    min_f1_score: float = 0.82
    max_p50_latency_ms: float = 50.0
    max_p95_latency_ms: float = 150.0
    max_p99_latency_ms: float = 300.0
    min_tps: float = 500.0  # transactions per second
    max_error_rate: float = 0.01  # 1%
    max_memory_mb: float = 512.0
    max_model_size_mb: float = 500.0
    min_uptime_ratio: float = 0.999


@dataclass
class ValidationResult:
    """Single validation check result."""
    name: str
    passed: bool
    actual_value: float
    threshold_value: float
    unit: str = ""
    details: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation report."""
    model_name: str
    model_version: str
    cluster_spec: str
    results: list = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ---------------------------------------------------------------------------
# Simulated model endpoint for demonstration
# ---------------------------------------------------------------------------
class SimulatedModelEndpoint:
    """Simulates a model serving endpoint for testing purposes.

    In production this would be replaced by HTTP calls to the actual
    model service running on EKS (e.g., via Seldon, KServe, or FastAPI).
    """

    def __init__(self, latency_mean_ms=30.0, latency_std_ms=15.0,
                 error_rate=0.005, accuracy=0.92):
        self.latency_mean_ms = latency_mean_ms
        self.latency_std_ms = latency_std_ms
        self.error_rate = error_rate
        self.accuracy = accuracy

    def predict(self, payload: dict) -> dict:
        latency_ms = max(1.0, np.random.normal(self.latency_mean_ms,
                                                self.latency_std_ms))
        time.sleep(latency_ms / 1000.0)

        if np.random.rand() < self.error_rate:
            raise RuntimeError("Simulated inference error")

        return {
            "prediction": int(np.random.randint(0, 3)),
            "confidence": float(np.random.uniform(0.7, 0.99)),
            "latency_ms": latency_ms,
        }

    def health(self) -> dict:
        return {"status": "healthy", "model_loaded": True}


# ---------------------------------------------------------------------------
# Validation suite
# ---------------------------------------------------------------------------
class ModelProductionValidator:
    """Run a suite of validation checks against a candidate model."""

    def __init__(self, endpoint: SimulatedModelEndpoint,
                 thresholds: ValidationThresholds,
                 cluster: EKSClusterSpec):
        self.endpoint = endpoint
        self.thresholds = thresholds
        self.cluster = cluster

    # -- Accuracy & quality checks -----------------------------------------
    def validate_accuracy(self, y_true, y_pred) -> ValidationResult:
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        accuracy = correct / len(y_true) if y_true else 0.0
        return ValidationResult(
            name="accuracy",
            passed=accuracy >= self.thresholds.min_accuracy,
            actual_value=round(accuracy, 4),
            threshold_value=self.thresholds.min_accuracy,
            details=f"Evaluated on {len(y_true)} samples",
        )

    def validate_f1_score(self, y_true, y_pred) -> ValidationResult:
        from collections import Counter

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length, "
                f"got {len(y_true)} vs {len(y_pred)}"
            )

        classes = sorted(set(y_true) | set(y_pred))
        f1_per_class = {}
        for cls in classes:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b == cls)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a != cls and b == cls)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b != cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            f1_per_class[cls] = f1

        class_counts = Counter(y_true)
        total = sum(class_counts.values())
        weighted_f1 = sum(
            f1_per_class[cls] * class_counts.get(cls, 0) / total
            for cls in classes
        )
        return ValidationResult(
            name="f1_score",
            passed=weighted_f1 >= self.thresholds.min_f1_score,
            actual_value=round(weighted_f1, 4),
            threshold_value=self.thresholds.min_f1_score,
        )

    # -- Latency & throughput checks ---------------------------------------
    def validate_latency(self, num_requests: int = 200) -> list:
        """Send sequential requests and measure latency distribution."""
        latencies = []
        errors = 0

        for _ in range(num_requests):
            try:
                payload = {"features": np.random.randn(10).tolist()}
                start = time.perf_counter()
                self.endpoint.predict(payload)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
            except Exception:
                errors += 1

        if not latencies:
            return [
                ValidationResult("p50_latency", False, 0, 0, "ms",
                                 "All requests failed"),
            ]

        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        error_rate = errors / num_requests

        return [
            ValidationResult(
                name="p50_latency",
                passed=p50 <= self.thresholds.max_p50_latency_ms,
                actual_value=round(p50, 2),
                threshold_value=self.thresholds.max_p50_latency_ms,
                unit="ms",
            ),
            ValidationResult(
                name="p95_latency",
                passed=p95 <= self.thresholds.max_p95_latency_ms,
                actual_value=round(p95, 2),
                threshold_value=self.thresholds.max_p95_latency_ms,
                unit="ms",
            ),
            ValidationResult(
                name="p99_latency",
                passed=p99 <= self.thresholds.max_p99_latency_ms,
                actual_value=round(p99, 2),
                threshold_value=self.thresholds.max_p99_latency_ms,
                unit="ms",
            ),
            ValidationResult(
                name="error_rate",
                passed=error_rate <= self.thresholds.max_error_rate,
                actual_value=round(error_rate, 4),
                threshold_value=self.thresholds.max_error_rate,
                details=f"{errors}/{num_requests} requests failed",
            ),
        ]

    def validate_tps(self, duration_seconds: int = 5,
                     concurrency: int = 10) -> ValidationResult:
        """Measure sustained transactions-per-second under concurrency."""
        completed = 0
        errors = 0
        counts_lock = threading.Lock()
        deadline = time.perf_counter() + duration_seconds

        def _send_one():
            nonlocal completed, errors
            while time.perf_counter() < deadline:
                try:
                    payload = {"features": np.random.randn(10).tolist()}
                    self.endpoint.predict(payload)
                    with counts_lock:
                        completed += 1
                except Exception:
                    with counts_lock:
                        errors += 1

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency) as pool:
            futures = [pool.submit(_send_one) for _ in range(concurrency)]
            concurrent.futures.wait(futures)

        actual_tps = completed / duration_seconds if duration_seconds else 0

        return ValidationResult(
            name="sustained_tps",
            passed=actual_tps >= self.thresholds.min_tps,
            actual_value=round(actual_tps, 1),
            threshold_value=self.thresholds.min_tps,
            unit="req/s",
            details=(f"{completed} ok, {errors} errors over "
                     f"{duration_seconds}s with concurrency={concurrency}"),
        )

    # -- Resource & sizing checks ------------------------------------------
    def validate_model_size(self, model_size_mb: float) -> ValidationResult:
        return ValidationResult(
            name="model_size",
            passed=model_size_mb <= self.thresholds.max_model_size_mb,
            actual_value=round(model_size_mb, 1),
            threshold_value=self.thresholds.max_model_size_mb,
            unit="MB",
        )

    def validate_memory_usage(self,
                              memory_usage_mb: float) -> ValidationResult:
        return ValidationResult(
            name="memory_usage",
            passed=memory_usage_mb <= self.thresholds.max_memory_mb,
            actual_value=round(memory_usage_mb, 1),
            threshold_value=self.thresholds.max_memory_mb,
            unit="MB",
        )

    def validate_eks_capacity(self, required_replicas: int,
                              cpu_per_replica: float,
                              memory_per_replica_gib: float) -> ValidationResult:
        """Check that the known EKS cluster can host the required replicas."""
        total_cpu_needed = required_replicas * cpu_per_replica
        total_mem_needed = required_replicas * memory_per_replica_gib

        available_cpu = self.cluster.total_vcpus * self.cluster.target_cpu_utilization
        available_mem = self.cluster.total_memory_gib * self.cluster.target_memory_utilization

        fits = (total_cpu_needed <= available_cpu
                and total_mem_needed <= available_mem)

        return ValidationResult(
            name="eks_capacity",
            passed=fits,
            actual_value=required_replicas,
            threshold_value=0,
            details=(f"Need {total_cpu_needed:.1f} vCPU / "
                     f"{total_mem_needed:.1f} GiB; "
                     f"cluster has {available_cpu:.1f} vCPU / "
                     f"{available_mem:.1f} GiB usable"),
        )

    # -- Health check ------------------------------------------------------
    def validate_health_endpoint(self) -> ValidationResult:
        try:
            resp = self.endpoint.health()
            healthy = resp.get("status") == "healthy"
            return ValidationResult(
                name="health_check",
                passed=healthy,
                actual_value=1.0 if healthy else 0.0,
                threshold_value=1.0,
                details=json.dumps(resp),
            )
        except Exception as exc:
            return ValidationResult(
                name="health_check",
                passed=False,
                actual_value=0.0,
                threshold_value=1.0,
                details=str(exc),
            )

    # -- Run all checks ----------------------------------------------------
    def run_full_suite(self, model_name: str, model_version: str,
                       y_true=None, y_pred=None,
                       model_size_mb: float = 120.0,
                       memory_usage_mb: float = 350.0,
                       required_replicas: int = 3,
                       cpu_per_replica: float = 2.0,
                       memory_per_replica_gib: float = 4.0) -> ValidationReport:
        """Execute the complete validation suite and return a report."""
        report = ValidationReport(
            model_name=model_name,
            model_version=model_version,
            cluster_spec=self.cluster.cluster_name,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Generate synthetic ground-truth if not provided
        if y_true is None or y_pred is None:
            rng = np.random.default_rng(42)
            y_true = rng.integers(0, 3, size=500).tolist()
            noise = rng.random(500)
            y_pred = [
                yt if n > (1 - self.endpoint.accuracy) else int((yt + 1) % 3)
                for yt, n in zip(y_true, noise)
            ]

        print("\n  [1/7] Checking health endpoint...")
        report.results.append(self.validate_health_endpoint())

        print("  [2/7] Evaluating accuracy...")
        report.results.append(self.validate_accuracy(y_true, y_pred))

        print("  [3/7] Evaluating F1 score...")
        report.results.append(self.validate_f1_score(y_true, y_pred))

        print("  [4/7] Measuring latency distribution...")
        report.results.extend(self.validate_latency(num_requests=100))

        print("  [5/7] Running sustained TPS load test...")
        report.results.append(
            self.validate_tps(duration_seconds=3, concurrency=8))

        print("  [6/7] Checking model size & memory...")
        report.results.append(self.validate_model_size(model_size_mb))
        report.results.append(self.validate_memory_usage(memory_usage_mb))

        print("  [7/7] Verifying EKS cluster capacity...")
        report.results.append(
            self.validate_eks_capacity(required_replicas, cpu_per_replica,
                                       memory_per_replica_gib))

        report.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return report


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def print_report(report: ValidationReport):
    """Pretty-print a validation report to stdout."""
    width = 78
    print("\n" + "=" * width)
    print("  MODEL PRODUCTION VALIDATION REPORT")
    print("=" * width)
    print(f"  Model:   {report.model_name} v{report.model_version}")
    print(f"  Cluster: {report.cluster_spec}")
    print(f"  Started: {report.started_at}")
    print(f"  Ended:   {report.finished_at}")
    print("-" * width)

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        unit = f" {r.unit}" if r.unit else ""
        line = (f"  [{status}] {r.name:<20s} "
                f"actual={r.actual_value}{unit:<8s} "
                f"threshold={r.threshold_value}{unit}")
        print(line)
        if r.details:
            print(f"         {r.details}")

    print("-" * width)
    verdict = "PROMOTED" if report.all_passed else "BLOCKED"
    print(f"  Result: {report.pass_count} passed, {report.fail_count} failed "
          f"=> {verdict}")
    print("=" * width)


def export_report_json(report: ValidationReport, path: Optional[str] = None):
    """Export the report as JSON for CI/CD consumption."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="validation_report_")
        os.close(fd)
    data = {
        "model_name": report.model_name,
        "model_version": report.model_version,
        "cluster_spec": report.cluster_spec,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "all_passed": report.all_passed,
        "results": [asdict(r) for r in report.results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Report exported to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("  Model Production Validation & Battle Testing")
    print("  Pre-deployment gate for Argo Rollouts promotion on EKS")
    print("=" * 78)

    # -- 1. Define EKS cluster spec ----------------------------------------
    cluster = EKSClusterSpec(
        cluster_name="ml-serving-prod",
        node_instance_type="m5.2xlarge",
        node_count=3,
    )
    print(f"\n  Target cluster: {cluster.cluster_name}")
    print(f"  Nodes: {cluster.node_count}x {cluster.node_instance_type} "
          f"({cluster.total_vcpus} vCPU, {cluster.total_memory_gib} GiB)")

    # -- 2. Define promotion thresholds ------------------------------------
    thresholds = ValidationThresholds(
        min_accuracy=0.85,
        min_f1_score=0.82,
        max_p50_latency_ms=50.0,
        max_p95_latency_ms=150.0,
        max_p99_latency_ms=300.0,
        min_tps=200.0,  # lowered for simulation demo
        max_error_rate=0.02,
        max_model_size_mb=500.0,
        max_memory_mb=512.0,
    )

    # -- 3. Spin up simulated model endpoint --------------------------------
    endpoint = SimulatedModelEndpoint(
        latency_mean_ms=25.0,
        latency_std_ms=12.0,
        error_rate=0.005,
        accuracy=0.92,
    )

    # -- 4. Run validation suite -------------------------------------------
    validator = ModelProductionValidator(endpoint, thresholds, cluster)

    print("\n  Running full validation suite...")
    report = validator.run_full_suite(
        model_name="customer-sentiment-classifier",
        model_version="2.1.0",
        model_size_mb=120.0,
        memory_usage_mb=350.0,
        required_replicas=3,
        cpu_per_replica=2.0,
        memory_per_replica_gib=4.0,
    )

    # -- 5. Print and export report ----------------------------------------
    print_report(report)
    export_report_json(report)

    # -- 6. Decide promotion -----------------------------------------------
    print("\n  Deployment Decision")
    print("  " + "-" * 40)
    if report.all_passed:
        print("  All checks passed.")
        print("  -> Trigger Argo Rollouts canary/blue-green promotion")
        print("  -> kubectl argo rollouts promote <rollout-name>")
    else:
        failed = [r.name for r in report.results if not r.passed]
        print(f"  Blocked on: {', '.join(failed)}")
        print("  -> Model stays in staging; rollout paused")
        print("  -> Fix issues and re-run validation")

    # -- Summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Production Validation Workflow")
    print("=" * 78)
    print("""
  1. Model trained & registered in MLflow/SageMaker Registry
  2. Model container built (BentoML / KServe / Seldon / FastAPI)
  3. Deployed to staging namespace on EKS
  4. THIS SCRIPT runs the validation suite:
     a. Accuracy & F1 against held-out test set
     b. Latency percentiles (p50/p95/p99)
     c. Sustained TPS under concurrent load
     d. Error rate under stress
     e. Model size & memory footprint
     f. EKS cluster capacity check
  5. If ALL checks pass -> Argo Rollouts promotion triggered
     - Canary: gradual traffic shift 10% -> 30% -> 60% -> 100%
     - Blue-Green: instant switch with automated rollback
  6. If ANY check fails -> rollout paused, team alerted

  See also:
    - code_folder/basics/kubernetes/argo-rollouts/  (K8s manifests)
    - code_folder/19_model_production_load_test.py  (detailed load harness)
    """)


if __name__ == "__main__":
    main()
