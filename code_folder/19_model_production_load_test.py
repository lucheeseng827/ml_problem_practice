"""
Model Production Load Test Harness
===================================
Category 19: MLOps - TPS / latency load testing for EKS-deployed models

Use cases: Pre-promotion TPS validation, soak tests, ramp-up stress tests
Demonstrates: Concurrent HTTP load generation, percentile tracking, EKS
              capacity estimation, go/no-go decision for Argo Rollouts

This script is meant to run as a Kubernetes Job or CI step that hits
the model's staging endpoint before Argo Rollouts promotion.
"""

import json
import os
import tempfile
import time
import threading
import statistics
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LoadTestConfig:
    """Configuration for a single load test run."""
    # Target endpoint (in real usage, the K8s service DNS)
    endpoint_url: str = "http://ml-model-canary-svc.ml-serving.svc.cluster.local/predict"
    # Ramp-up phases: list of (concurrency, duration_seconds)
    phases: list = field(default_factory=lambda: [
        (5, 10),     # warm-up: 5 workers for 10s
        (10, 15),    # ramp: 10 workers for 15s
        (25, 20),    # target: 25 workers for 20s
        (50, 20),    # peak: 50 workers for 20s
        (10, 10),    # cool-down: 10 workers for 10s
    ])
    # Pass/fail thresholds
    min_tps: float = 500.0
    max_p50_latency_ms: float = 50.0
    max_p95_latency_ms: float = 150.0
    max_p99_latency_ms: float = 300.0
    max_error_rate: float = 0.01


@dataclass
class PhaseResult:
    """Metrics captured during a single load phase."""
    phase_index: int
    concurrency: int
    duration_seconds: int
    total_requests: int = 0
    total_errors: int = 0
    latencies_ms: list = field(default_factory=list)

    @property
    def tps(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds else 0

    @property
    def error_rate(self) -> float:
        total = self.total_requests + self.total_errors
        return self.total_errors / total if total else 0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, p))


# ---------------------------------------------------------------------------
# Simulated HTTP client (replace with requests/httpx in real usage)
# ---------------------------------------------------------------------------
class SimulatedHTTPClient:
    """Simulates HTTP POST to the model endpoint.

    In production, replace with:
        import httpx
        resp = httpx.post(url, json=payload, timeout=5.0)
    """

    def __init__(self, latency_mean_ms=30.0, latency_std_ms=15.0,
                 error_rate=0.005):
        self.latency_mean_ms = latency_mean_ms
        self.latency_std_ms = latency_std_ms
        self.error_rate = error_rate

    def post(self, url: str, payload: dict) -> dict:
        latency_ms = max(1.0, np.random.normal(self.latency_mean_ms,
                                                self.latency_std_ms))
        time.sleep(latency_ms / 1000.0)

        if np.random.rand() < self.error_rate:
            raise ConnectionError("Simulated 5xx from model endpoint")

        return {"prediction": 1, "confidence": 0.91, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Load test runner
# ---------------------------------------------------------------------------
class LoadTestRunner:
    """Execute a multi-phase load test against a model endpoint."""

    def __init__(self, config: LoadTestConfig, client=None):
        self.config = config
        self.client = client or SimulatedHTTPClient()

    def _worker(self, deadline: float, url: str, results: dict,
                lock: threading.Lock):
        """Single worker loop: send requests until deadline."""
        while time.perf_counter() < deadline:
            payload = {"features": np.random.randn(10).tolist()}
            start = time.perf_counter()
            try:
                self.client.post(url, payload)
                elapsed_ms = (time.perf_counter() - start) * 1000
                with lock:
                    results["latencies"].append(elapsed_ms)
                    results["ok"] += 1
            except Exception as e:
                with lock:
                    results["errors"] += 1
                    results.setdefault("last_error", str(e))

    def run_phase(self, phase_index: int, concurrency: int,
                  duration: int) -> PhaseResult:
        """Run a single load phase."""
        shared = {"latencies": [], "ok": 0, "errors": 0}
        shared_lock = threading.Lock()
        deadline = time.perf_counter() + duration

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency) as pool:
            futures = [
                pool.submit(self._worker, deadline,
                            self.config.endpoint_url, shared,
                            shared_lock)
                for _ in range(concurrency)
            ]
            concurrent.futures.wait(futures)

        return PhaseResult(
            phase_index=phase_index,
            concurrency=concurrency,
            duration_seconds=duration,
            total_requests=shared["ok"],
            total_errors=shared["errors"],
            latencies_ms=shared["latencies"],
        )

    def run(self) -> list:
        """Execute all phases sequentially and return results."""
        results = []
        for i, (conc, dur) in enumerate(self.config.phases):
            print(f"  Phase {i+1}/{len(self.config.phases)}: "
                  f"concurrency={conc}, duration={dur}s")
            pr = self.run_phase(i, conc, dur)
            print(f"    -> {pr.total_requests} ok, {pr.total_errors} errors, "
                  f"TPS={pr.tps:.1f}, "
                  f"p50={pr.percentile(50):.1f}ms, "
                  f"p95={pr.percentile(95):.1f}ms, "
                  f"p99={pr.percentile(99):.1f}ms")
            results.append(pr)
        return results


# ---------------------------------------------------------------------------
# Evaluation / go-no-go
# ---------------------------------------------------------------------------
def evaluate_results(phases: list, config: LoadTestConfig) -> dict:
    """Evaluate load test results against thresholds.

    Returns a dict with pass/fail per metric and overall verdict.
    """
    # Use the "target" phase (highest concurrency that is not cool-down)
    # to evaluate thresholds.  Fallback to the last phase.
    target_phase = max(phases, key=lambda p: p.concurrency)

    checks = {
        "tps": {
            "actual": round(target_phase.tps, 1),
            "threshold": config.min_tps,
            "passed": target_phase.tps >= config.min_tps,
        },
        "p50_latency_ms": {
            "actual": round(target_phase.percentile(50), 2),
            "threshold": config.max_p50_latency_ms,
            "passed": target_phase.percentile(50) <= config.max_p50_latency_ms,
        },
        "p95_latency_ms": {
            "actual": round(target_phase.percentile(95), 2),
            "threshold": config.max_p95_latency_ms,
            "passed": target_phase.percentile(95) <= config.max_p95_latency_ms,
        },
        "p99_latency_ms": {
            "actual": round(target_phase.percentile(99), 2),
            "threshold": config.max_p99_latency_ms,
            "passed": target_phase.percentile(99) <= config.max_p99_latency_ms,
        },
        "error_rate": {
            "actual": round(target_phase.error_rate, 4),
            "threshold": config.max_error_rate,
            "passed": target_phase.error_rate <= config.max_error_rate,
        },
    }

    all_passed = all(c["passed"] for c in checks.values())
    return {"checks": checks, "all_passed": all_passed,
            "peak_phase_concurrency": target_phase.concurrency}


def print_evaluation(evaluation: dict):
    """Pretty-print the go/no-go evaluation."""
    width = 70
    print("\n" + "=" * width)
    print("  LOAD TEST EVALUATION")
    print("=" * width)

    for name, check in evaluation["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name:<20s} "
              f"actual={check['actual']:<10} threshold={check['threshold']}")

    print("-" * width)
    verdict = "GO - promote rollout" if evaluation["all_passed"] else \
              "NO-GO - rollout blocked"
    print(f"  Verdict: {verdict}")
    print(f"  Evaluated at peak concurrency: "
          f"{evaluation['peak_phase_concurrency']}")
    print("=" * width)


def export_results(phases: list, evaluation: dict,
                   path: Optional[str] = None):
    """Export results as JSON for CI/CD pipelines."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="load_test_results_")
        os.close(fd)
    data = {
        "phases": [
            {
                "phase_index": p.phase_index,
                "concurrency": p.concurrency,
                "duration_seconds": p.duration_seconds,
                "total_requests": p.total_requests,
                "total_errors": p.total_errors,
                "tps": round(p.tps, 1),
                "p50_ms": round(p.percentile(50), 2),
                "p95_ms": round(p.percentile(95), 2),
                "p99_ms": round(p.percentile(99), 2),
                "error_rate": round(p.error_rate, 4),
            }
            for p in phases
        ],
        "evaluation": evaluation,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results exported to {path}")


# ---------------------------------------------------------------------------
# Kubernetes Job manifest helper
# ---------------------------------------------------------------------------
def _validate_k8s_value(value: str, name: str) -> str:
    """Reject values that could break YAML structure."""
    forbidden = set('\n\r\t\x00')
    if any(c in forbidden for c in value):
        raise ValueError(
            f"{name} contains invalid characters (newlines, tabs, or nulls)"
        )
    return value


def generate_k8s_job_manifest(image: str, endpoint_url: str,
                              namespace: str = "ml-serving") -> str:
    """Generate a K8s Job manifest to run this load test on the cluster."""
    image = _validate_k8s_value(image, "image")
    endpoint_url = _validate_k8s_value(endpoint_url, "endpoint_url")
    namespace = _validate_k8s_value(namespace, "namespace")

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "ml-model-load-test",
            "namespace": namespace,
            "labels": {"app": "ml-model-load-test"},
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "load-tester",
                        "image": image,
                        "command": ["python", "19_model_production_load_test.py"],
                        "env": [
                            {"name": "ENDPOINT_URL", "value": endpoint_url},
                            {"name": "MIN_TPS", "value": "500"},
                            {"name": "MAX_P95_LATENCY_MS", "value": "150"},
                        ],
                        "resources": {
                            "requests": {"cpu": "1", "memory": "512Mi"},
                            "limits": {"cpu": "2", "memory": "1Gi"},
                        },
                    }],
                }
            },
        },
    }
    return json.dumps(manifest, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Model Production Load Test Harness")
    print("  TPS & Latency validation for EKS-deployed model endpoints")
    print("=" * 70)

    # -- Configure test (in CI, read from env vars) -------------------------
    config = LoadTestConfig(
        endpoint_url="http://ml-model-canary-svc.ml-serving.svc.cluster.local/predict",
        phases=[
            (2, 3),    # warm-up
            (5, 5),    # ramp
            (10, 5),   # target
            (15, 5),   # peak
            (5, 3),    # cool-down
        ],
        min_tps=100.0,       # lowered for demo simulation
        max_p50_latency_ms=60.0,
        max_p95_latency_ms=150.0,
        max_p99_latency_ms=300.0,
        max_error_rate=0.02,
    )

    print(f"\n  Target endpoint: {config.endpoint_url}")
    print(f"  Phases: {len(config.phases)}")
    print(f"  Thresholds: min_tps={config.min_tps}, "
          f"p95<={config.max_p95_latency_ms}ms, "
          f"error_rate<={config.max_error_rate}")

    # -- Run load test ------------------------------------------------------
    print("\n  Starting load test...\n")
    runner = LoadTestRunner(config)
    phases = runner.run()

    # -- Evaluate -----------------------------------------------------------
    evaluation = evaluate_results(phases, config)
    print_evaluation(evaluation)
    export_results(phases, evaluation)

    # -- Show K8s Job manifest example --------------------------------------
    print("\n  Example: Run as Kubernetes Job on EKS")
    print("  " + "-" * 50)
    manifest = generate_k8s_job_manifest(
        image="registry.example.com/ml-load-tester:latest",
        endpoint_url=config.endpoint_url,
    )
    for line in manifest.split("\n"):
        print(f"  {line}")

    # -- Integration with Argo Rollouts ------------------------------------
    print("\n" + "=" * 70)
    print("  Integration with Argo Rollouts")
    print("=" * 70)
    print("""
  This load test integrates into the Argo Rollouts promotion workflow:

  1. New model image pushed to registry
  2. Argo Rollout creates canary/preview pods on EKS
  3. Load test Job launched against canary/preview Service
  4. If load test passes (all_passed=true):
       kubectl argo rollouts promote ml-model-canary
  5. If load test fails:
       kubectl argo rollouts abort ml-model-canary
       (automatic rollback)

  For canary: load test runs at each traffic step (10%, 30%, 60%)
  For blue-green: load test runs against preview before promotion

  CI Pipeline Example (GitHub Actions):
    - name: Run load test
      run: |
        kubectl apply -f load-test-job.yaml
        kubectl wait --for=condition=complete job/ml-model-load-test -n ml-serving
        RESULT=$(kubectl logs job/ml-model-load-test -n ml-serving | tail -1)
        if echo "$RESULT" | grep -q "NO-GO"; then
          kubectl argo rollouts abort ml-model-canary -n ml-serving
          exit 1
        fi
        kubectl argo rollouts promote ml-model-canary -n ml-serving

  See also:
    - code_folder/basics/kubernetes/argo-rollouts/canary-rollout.yaml
    - code_folder/basics/kubernetes/argo-rollouts/blue-green-rollout.yaml
    - code_folder/19_model_production_validation.py
    """)


if __name__ == "__main__":
    main()
