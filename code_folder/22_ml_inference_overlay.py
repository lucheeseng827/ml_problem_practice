"""
Category 22: ML Inference & Overlay on Test Dataset
=====================================================
Load trained model, run inference on held-out test data,
overlay predictions on the test dataframe, and perform error analysis.

Usage:
    python 22_ml_inference_overlay.py

SageMaker Batch Transform:
    Model:  /opt/ml/model/
    Input:  test CSV from S3
    Output: predictions CSV to S3
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_data_dir() -> Path:
    """Return data directory for test data."""
    if os.path.isdir("/opt/ml/processing/input"):
        return Path("/opt/ml/processing/input")
    return Path(__file__).parent.parent / "data" / "engineered"


def get_model_dir() -> Path:
    """Return model artifacts directory."""
    if os.path.isdir("/opt/ml/model"):
        return Path("/opt/ml/model")
    return Path(__file__).parent.parent / "model_artifacts"


def get_output_dir() -> Path:
    """Return output directory for predictions."""
    if os.path.isdir("/opt/ml"):
        output_dir = Path("/opt/ml/processing/output")
    else:
        output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# 1. Load Test Data and Model
# ---------------------------------------------------------------------------
def load_test_data(data_dir: Path, target_col: str = "Survived"):
    """Load held-out test dataset."""
    test_path = data_dir / "test.csv"
    print(f"[Inference] Loading test data from {test_path}")
    df = pd.read_csv(test_path)

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    print(f"  -> X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_test, y_test


def load_model(model_dir: Path, model_name: str = "xgb_model.joblib"):
    """Load a trained model from artifacts."""
    model_path = model_dir / model_name
    print(f"[Inference] Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"  -> Model type: {type(model).__name__}")
    return model


# ---------------------------------------------------------------------------
# 2. Run Inference
# ---------------------------------------------------------------------------
def run_inference(model, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities."""
    print(f"[Inference] Running predictions on {X_test.shape[0]} samples")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"  -> Predictions generated")
    print(f"  -> Predicted class distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    return y_pred, y_proba


# ---------------------------------------------------------------------------
# 3. Overlay Predictions on Test DataFrame
# ---------------------------------------------------------------------------
def overlay_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """Overlay model predictions onto the test dataframe."""
    print("[Overlay] Adding predictions to test dataframe")
    result = X_test.copy()
    result["Actual"] = y_test.values
    result["Predicted"] = y_pred
    result["Predicted_Probability"] = y_proba
    result["Correct"] = (result["Actual"] == result["Predicted"]).astype(int)
    result["Confidence"] = np.where(
        result["Predicted"] == 1,
        result["Predicted_Probability"],
        1 - result["Predicted_Probability"],
    )

    print(f"  -> Overlay dataframe shape: {result.shape}")
    print(f"  -> Columns added: Actual, Predicted, Predicted_Probability, Correct, Confidence")
    return result


# ---------------------------------------------------------------------------
# 4. Evaluate Metrics
# ---------------------------------------------------------------------------
def evaluate_metrics(y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute full evaluation metrics on test set."""
    print("\n" + "=" * 60)
    print("[Evaluate] Test Set Metrics")
    print("=" * 60)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"               Pred=0  Pred=1")
    print(f"    Actual=0   {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"    Actual=1   {cm[1][0]:>5}   {cm[1][1]:>5}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, indent=4))

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


# ---------------------------------------------------------------------------
# 5. Error Analysis
# ---------------------------------------------------------------------------
def error_analysis(overlay_df: pd.DataFrame) -> dict:
    """Analyze prediction errors by feature segments."""
    print("\n" + "=" * 60)
    print("[Error Analysis]")
    print("=" * 60)

    errors = overlay_df[overlay_df["Correct"] == 0]
    correct = overlay_df[overlay_df["Correct"] == 1]
    total = len(overlay_df)
    n_errors = len(errors)

    print(f"  Total samples:  {total}")
    print(f"  Correct:        {len(correct)} ({len(correct)/total:.1%})")
    print(f"  Errors:         {n_errors} ({n_errors/total:.1%})")

    # Error distribution by actual class
    print(f"\n  Errors by actual class:")
    for cls, count in errors["Actual"].value_counts().items():
        total_cls = (overlay_df["Actual"] == cls).sum()
        print(f"    Class {cls}: {count} errors / {total_cls} total ({count/total_cls:.1%})")

    # Confidence distribution for errors vs correct
    print(f"\n  Confidence stats:")
    print(f"    Correct predictions - mean confidence: {correct['Confidence'].mean():.4f}")
    print(f"    Wrong predictions   - mean confidence: {errors['Confidence'].mean():.4f}")

    # Low-confidence predictions (potential uncertainty zone)
    low_conf = overlay_df[overlay_df["Confidence"] < 0.6]
    if len(low_conf) > 0:
        low_conf_errors = low_conf[low_conf["Correct"] == 0]
        print(f"\n  Low confidence (<0.6) predictions: {len(low_conf)}")
        print(f"    Of which incorrect: {len(low_conf_errors)} ({len(low_conf_errors)/len(low_conf):.1%})")

    analysis = {
        "total_samples": total,
        "total_errors": n_errors,
        "error_rate": n_errors / total,
        "mean_confidence_correct": float(correct["Confidence"].mean()),
        "mean_confidence_errors": float(errors["Confidence"].mean()) if n_errors > 0 else None,
    }
    return analysis


# ---------------------------------------------------------------------------
# 6. Export Results
# ---------------------------------------------------------------------------
def export_results(
    overlay_df: pd.DataFrame,
    metrics: dict,
    error_stats: dict,
    output_dir: Path,
) -> None:
    """Save overlay predictions and metrics to files."""
    print(f"\n[Export] Saving results to {output_dir}")

    # Full overlay
    overlay_df.to_csv(output_dir / "test_predictions_overlay.csv", index=False)
    overlay_df.to_parquet(output_dir / "test_predictions_overlay.parquet", index=False)
    print(f"  -> test_predictions_overlay.csv ({overlay_df.shape[0]} rows)")

    # Errors only
    errors_df = overlay_df[overlay_df["Correct"] == 0]
    errors_df.to_csv(output_dir / "test_errors.csv", index=False)
    print(f"  -> test_errors.csv ({errors_df.shape[0]} rows)")

    # Metrics
    results = {"metrics": metrics, "error_analysis": error_stats}
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  -> test_results.json")

    # Summary table
    summary = pd.DataFrame([metrics]).drop(columns=["confusion_matrix"], errors="ignore")
    summary.to_csv(output_dir / "test_metrics_summary.csv", index=False)
    print(f"  -> test_metrics_summary.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_dir = get_data_dir()
    model_dir = get_model_dir()
    output_dir = get_output_dir()

    # Load
    X_test, y_test = load_test_data(data_dir)
    model = load_model(model_dir, "xgb_model.joblib")

    # Inference
    y_pred, y_proba = run_inference(model, X_test)

    # Overlay predictions on test dataframe
    overlay_df = overlay_predictions(X_test, y_test, y_pred, y_proba)

    # Evaluate
    metrics = evaluate_metrics(y_test, y_pred, y_proba)

    # Error analysis
    error_stats = error_analysis(overlay_df)

    # Export
    export_results(overlay_df, metrics, error_stats, output_dir)

    print("\n[Done] Inference and overlay complete.")
    return overlay_df, metrics


if __name__ == "__main__":
    main()
