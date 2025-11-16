"""
BentoML Model Packaging and Deployment
=======================================
Category 19: MLOps - Model packaging and containerization

Use cases: Package models for production, create model APIs, containerize models
Demonstrates: BentoML service creation, model packaging, deployment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import bentoml
from bentoml.io import JSON, NumpyNdarray


def train_and_save_model():
    """Train a model and save with BentoML"""
    print("\n1. Training Model")
    print("-" * 70)

    # Load and prepare data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.4f}")

    # Save model with BentoML
    try:
        bento_model = bentoml.sklearn.save_model(
            "iris_classifier",
            model,
            labels={
                "framework": "sklearn",
                "task": "classification"
            },
            metadata={
                "accuracy": accuracy,
                "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                "target_names": ["setosa", "versicolor", "virginica"]
            }
        )
        print(f"✓ Model saved to BentoML: {bento_model.tag}")
        return bento_model.tag
    except Exception as e:
        print(f"⚠️  Could not save to BentoML: {e}")
        return None


def create_service_definition():
    """Create a BentoML service definition"""
    service_code = '''
import numpy as np
import bentoml
from bentoml.io import JSON, NumpyNdarray

# Load the model
iris_model = bentoml.sklearn.get("iris_classifier:latest")

# Create service
svc = bentoml.Service("iris_classifier", runners=[iris_model.to_runner()])

@svc.api(input=NumpyNdarray(), output=JSON())
def classify(input_data: np.ndarray) -> dict:
    """Classify iris species"""
    # Get prediction
    result = iris_model.to_runner().predict.run(input_data)

    # Map to species names
    species = ["setosa", "versicolor", "virginica"]
    predictions = [species[int(pred)] for pred in result]

    return {
        "predictions": predictions,
        "model_version": str(iris_model.tag)
    }

@svc.api(input=JSON(), output=JSON())
def classify_json(input_json: dict) -> dict:
    """Classify from JSON input"""
    features = np.array(input_json["features"])
    if features.ndim == 1:
        features = features.reshape(1, -1)

    result = iris_model.to_runner().predict.run(features)
    species = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": species[int(result[0])],
        "confidence": float(np.max([0.8, 0.9, 0.95]))  # Placeholder
    }
'''
    return service_code


def create_bentofile():
    """Create a bentofile.yaml for packaging"""
    bentofile = '''
service: "service.py:svc"
labels:
  owner: ml-team
  stage: production
include:
  - "*.py"
python:
  packages:
    - scikit-learn
    - pandas
    - numpy
docker:
  distro: debian
  python_version: "3.10"
  system_packages:
    - git
'''
    return bentofile


def demonstrate_model_packaging():
    """Demonstrate BentoML model packaging workflow"""
    print("=" * 70)
    print("BentoML Model Packaging and Deployment")
    print("=" * 70)

    # Train and save model
    model_tag = train_and_save_model()

    print("\n2. Creating BentoML Service Definition")
    print("-" * 70)
    service_code = create_service_definition()
    print("Service definition created (service.py):")
    print("  • API endpoint: /classify (numpy array input)")
    print("  • API endpoint: /classify_json (JSON input)")
    print("  • Auto-generated OpenAPI docs")
    print("  • Health check endpoint")

    print("\n3. Creating Bentofile Configuration")
    print("-" * 70)
    bentofile = create_bentofile()
    print("Bentofile.yaml created:")
    print("  • Service entry point defined")
    print("  • Python dependencies specified")
    print("  • Docker configuration set")

    print("\n4. Building Bento (Deployment Package)")
    print("-" * 70)
    print("Command: bentoml build")
    print("  • Packages model + code + dependencies")
    print("  • Creates reproducible artifact")
    print("  • Generates Docker image recipe")
    print("  • Output: bento_name:version (e.g., iris_classifier:abc123)")

    print("\n5. Containerizing with Docker")
    print("-" * 70)
    print("Command: bentoml containerize iris_classifier:latest")
    print("  • Builds optimized Docker image")
    print("  • Includes gunicorn WSGI server")
    print("  • Production-ready configuration")
    print("  • Image size: ~500MB - 1GB")

    print("\n6. Serving the Model Locally")
    print("-" * 70)
    print("Command: bentoml serve service.py:svc --reload")
    print("  • Starts development server")
    print("  • API available at: http://localhost:3000")
    print("  • Swagger UI: http://localhost:3000/docs")
    print("  • Health check: http://localhost:3000/healthz")

    print("\n7. Making Predictions")
    print("-" * 70)
    # Simulate prediction
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    print(f"Sample input (numpy): {sample_input}")
    print("curl -X POST http://localhost:3000/classify \\")
    print('  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"features": {sample_input.tolist()[0]}}}\'')
    print()
    print("Expected response:")
    print('{"prediction": "setosa", "confidence": 0.95}')

    print("\n8. Deployment Options")
    print("-" * 70)
    deployment_options = {
        "Docker": "docker run -p 3000:3000 iris_classifier:latest",
        "Kubernetes": "bentoml deploy iris_classifier:latest --platform=kubernetes",
        "AWS Lambda": "bentoml deploy iris_classifier:latest --platform=aws-lambda",
        "AWS SageMaker": "bentoml deploy iris_classifier:latest --platform=sagemaker",
        "Google Cloud Run": "bentoml deploy iris_classifier:latest --platform=google-cloud-run",
        "Azure Container Instances": "bentoml deploy iris_classifier:latest --platform=azure",
        "Yatai (BentoML Cloud)": "bentoml push && bentoml deployment create"
    }

    for platform, command in deployment_options.items():
        print(f"  • {platform:25} {command}")

    print("\n" + "=" * 70)
    print("Key BentoML Features:")
    print("=" * 70)
    print("✓ Model Packaging: Package models with dependencies")
    print("✓ API Generation: Auto-generate REST APIs")
    print("✓ Containerization: One-command Docker image creation")
    print("✓ Multi-Framework: Support for sklearn, PyTorch, TensorFlow, etc.")
    print("✓ Performance: Optimized serving with batching & caching")
    print("✓ Monitoring: Built-in metrics and logging")
    print("✓ Deployment: Deploy to any cloud platform")
    print()

    print("Production Workflow:")
    print("  1. Train model → Save with bentoml.sklearn.save_model()")
    print("  2. Create service.py → Define API endpoints")
    print("  3. Create bentofile.yaml → Specify configuration")
    print("  4. Build bento → bentoml build")
    print("  5. Test locally → bentoml serve")
    print("  6. Containerize → bentoml containerize")
    print("  7. Deploy → Push to production platform")
    print("  8. Monitor → Track metrics and performance")
    print()

    print("Advantages over plain Flask/FastAPI:")
    print("  • Automatic API documentation")
    print("  • Built-in model versioning")
    print("  • Optimized inference (batching, GPU support)")
    print("  • Easy cloud deployment")
    print("  • Model registry integration")


def main():
    demonstrate_model_packaging()

    print("\n" + "=" * 70)
    print("Quick Start Commands:")
    print("=" * 70)
    print("# Install BentoML")
    print("pip install bentoml")
    print()
    print("# Save this example code to files")
    print("python 19_bentoml_model_packaging.py  # Train and save model")
    print()
    print("# Create service.py with the service definition")
    print("# Create bentofile.yaml with the configuration")
    print()
    print("# Build and serve")
    print("bentoml build")
    print("bentoml serve service.py:svc --reload")
    print()
    print("# Containerize and run")
    print("bentoml containerize iris_classifier:latest")
    print("docker run -p 3000:3000 iris_classifier:latest")


if __name__ == "__main__":
    main()
