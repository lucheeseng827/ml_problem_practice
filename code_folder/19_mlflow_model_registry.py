"""
MLflow Model Registry and Versioning
=====================================
Category 19: MLOps - Model versioning and lifecycle management

Use cases: Model versioning, staging, production deployment, rollback
Demonstrates: MLflow tracking, model registry, model stages
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def train_model(n_estimators=100, max_depth=10):
    """Train a model with MLflow tracking"""
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return model, accuracy, f1


def main():
    print("=" * 70)
    print("MLflow Model Registry - Model Versioning & Lifecycle Management")
    print("=" * 70)

    # Set MLflow tracking URI (use local or remote server)
    mlflow.set_tracking_uri("http://localhost:5000")  # Change to your MLflow server
    mlflow.set_experiment("iris_classification")

    model_name = "iris_classifier"

    print("\n1. Training Multiple Model Versions")
    print("-" * 70)

    # Train multiple versions with different hyperparameters
    configs = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ]

    best_run_id = None
    best_accuracy = 0

    for i, config in enumerate(configs):
        with mlflow.start_run(run_name=f"version_{i+1}"):
            print(f"\nTraining version {i+1} with config: {config}")

            # Log parameters
            mlflow.log_params(config)

            # Train model
            model, accuracy, f1 = train_model(**config)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score": f1
            })

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Run ID: {mlflow.active_run().info.run_id}")

            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = mlflow.active_run().info.run_id

    print(f"\n✓ Best model: Run ID {best_run_id} with accuracy {best_accuracy:.4f}")

    print("\n2. Model Registry - Managing Model Stages")
    print("-" * 70)

    try:
        client = mlflow.tracking.MlflowClient()

        # Get all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        print(f"\nFound {len(model_versions)} versions of '{model_name}'")

        if model_versions:
            latest_version = model_versions[0]
            version_number = latest_version.version

            print(f"\n3. Transitioning Model to Staging")
            print("-" * 70)
            # Transition to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=version_number,
                stage="Staging"
            )
            print(f"✓ Version {version_number} moved to 'Staging'")

            print(f"\n4. Loading Model from Registry")
            print("-" * 70)
            # Load model from staging
            model_uri = f"models:/{model_name}/Staging"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            print(f"✓ Loaded model from: {model_uri}")

            # Test loaded model
            X_test = np.array([[5.1, 3.5, 1.4, 0.2]])
            prediction = loaded_model.predict(X_test)
            print(f"\nTest prediction: {prediction[0]}")

            print(f"\n5. Promoting Model to Production")
            print("-" * 70)
            # After validation, promote to Production
            client.transition_model_version_stage(
                name=model_name,
                version=version_number,
                stage="Production"
            )
            print(f"✓ Version {version_number} promoted to 'Production'")

            print(f"\n6. Model Versioning Information")
            print("-" * 70)
            # Get model version details
            version_details = client.get_model_version(model_name, version_number)
            print(f"Model: {version_details.name}")
            print(f"Version: {version_details.version}")
            print(f"Stage: {version_details.current_stage}")
            print(f"Run ID: {version_details.run_id}")
            print(f"Created: {version_details.creation_timestamp}")

    except Exception as e:
        print(f"\n⚠️  MLflow Registry operation failed: {e}")
        print("Note: Ensure MLflow tracking server is running:")
        print("  mlflow server --host 0.0.0.0 --port 5000")

    print("\n" + "=" * 70)
    print("Key MLOps Practices:")
    print("=" * 70)
    print("✓ Experiment Tracking: Track all experiments with parameters and metrics")
    print("✓ Model Versioning: Automatic versioning of all trained models")
    print("✓ Model Stages: None → Staging → Production → Archived")
    print("✓ Model Registry: Centralized repository for model artifacts")
    print("✓ Reproducibility: Every model linked to exact training code/data")
    print("✓ Governance: Audit trail of model changes and promotions")
    print()

    print("Model Lifecycle Stages:")
    print("  • None: Newly registered model")
    print("  • Staging: Model being tested/validated")
    print("  • Production: Model serving live traffic")
    print("  • Archived: Deprecated model")
    print()

    print("Production Workflow:")
    print("  1. Train multiple model versions")
    print("  2. Register best model to MLflow Registry")
    print("  3. Transition to 'Staging' for validation")
    print("  4. Run A/B tests and performance evaluation")
    print("  5. Promote to 'Production' if metrics improve")
    print("  6. Monitor production model performance")
    print("  7. Archive old versions when superseded")


if __name__ == "__main__":
    main()
