"""
Category 22: ML Model Training
================================
Train simple ML models on engineered Titanic data.

Models:
  - Logistic Regression (baseline)
  - XGBoost Classifier
  - Cross-validation for both

Saves model artifacts for inference stage.

Usage:
    python 22_ml_model_training.py

SageMaker Training Job:
    Input channels:  train -> /opt/ml/input/data/train/
                     test  -> /opt/ml/input/data/test/
    Model output:    /opt/ml/model/
    Hyperparameters: n-estimators, max-depth, learning-rate
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def get_data_dir() -> Path:
    """Return data directory — uses SageMaker channels if available."""
    if os.path.isdir("/opt/ml/input/data/train"):
        return Path("/opt/ml/input/data/train")
    return Path(__file__).parent.parent / "data" / "engineered"


def get_model_dir() -> Path:
    """Return model output directory."""
    if os.path.isdir("/opt/ml/model"):
        model_dir = Path("/opt/ml/model")
    else:
        model_dir = Path(__file__).parent.parent / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def parse_args():
    """Parse hyperparameters — from CLI or SageMaker hyperparameter JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--cv-folds", type=int, default=5)
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
def load_train_data(data_dir: Path, target_col: str = "Survived"):
    """Load training data and split features/target."""
    train_path = data_dir / "train.csv"
    print(f"[Train] Loading training data from {train_path}")
    df = pd.read_csv(train_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"  -> X: {X.shape}, y: {y.shape}")
    print(f"  -> Target distribution: {y.value_counts(normalize=True).to_dict()}")
    return X, y


# ---------------------------------------------------------------------------
# Train Logistic Regression (Baseline)
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train, cv_folds: int = 5):
    """Train and evaluate a Logistic Regression model."""
    print("\n" + "=" * 60)
    print("[Model 1] Logistic Regression (Baseline)")
    print("=" * 60)

    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train, y_train)

    # Training metrics
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    print(f"  Train Accuracy:  {accuracy_score(y_train, y_pred):.4f}")
    print(f"  Train ROC-AUC:   {roc_auc_score(y_train, y_proba):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"  CV Accuracy:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    cv_auc = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="roc_auc")
    print(f"  CV ROC-AUC:      {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    return model, {
        "model_type": "LogisticRegression",
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "train_roc_auc": float(roc_auc_score(y_train, y_proba)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
    }


# ---------------------------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train, args, cv_folds: int = 5):
    """Train and evaluate an XGBoost model."""
    print("\n" + "=" * 60)
    print("[Model 2] XGBoost Classifier")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # Training metrics
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    print(f"  Train Accuracy:  {accuracy_score(y_train, y_pred):.4f}")
    print(f"  Train ROC-AUC:   {roc_auc_score(y_train, y_proba):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"  CV Accuracy:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    cv_auc = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="roc_auc")
    print(f"  CV ROC-AUC:      {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    # Feature importance
    importance = dict(
        sorted(
            zip(X_train.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    print("  Top 5 features:")
    for feat, imp in list(importance.items())[:5]:
        print(f"    {feat}: {imp:.4f}")

    return model, {
        "model_type": "XGBClassifier",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "train_roc_auc": float(roc_auc_score(y_train, y_proba)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "feature_importance": {k: float(v) for k, v in importance.items()},
    }


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------
def compare_models(lr_metrics: dict, xgb_metrics: dict) -> str:
    """Compare models and pick the best based on CV ROC-AUC."""
    print("\n" + "=" * 60)
    print("[Compare] Model Comparison")
    print("=" * 60)

    print(f"  {'Metric':<25} {'LogReg':>10} {'XGBoost':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for metric in ["train_accuracy", "train_roc_auc", "cv_accuracy_mean", "cv_roc_auc_mean"]:
        lr_val = lr_metrics[metric]
        xgb_val = xgb_metrics[metric]
        marker = " <-" if xgb_val > lr_val else ""
        print(f"  {metric:<25} {lr_val:>10.4f} {xgb_val:>10.4f}{marker}")

    best = "xgboost" if xgb_metrics["cv_roc_auc_mean"] > lr_metrics["cv_roc_auc_mean"] else "logistic"
    print(f"\n  Best model (CV ROC-AUC): {best}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data_dir = get_data_dir()
    model_dir = get_model_dir()

    X_train, y_train = load_train_data(data_dir)

    # Train both models
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, args.cv_folds)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, args, args.cv_folds)

    # Compare
    best_model_name = compare_models(lr_metrics, xgb_metrics)

    # Save artifacts
    joblib.dump(lr_model, model_dir / "logistic_model.joblib")
    joblib.dump(xgb_model, model_dir / "xgb_model.joblib")
    print(f"\n[Save] Models saved to {model_dir}")

    # Save training metadata
    training_metadata = {
        "best_model": best_model_name,
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "features": list(X_train.columns),
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0],
    }
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(training_metadata, f, indent=2)

    print(f"[Done] Training complete. Best model: {best_model_name}")
    return lr_model, xgb_model, training_metadata


if __name__ == "__main__":
    main()
